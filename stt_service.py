"""STT orchestration: catalog attempt hierarchy + OpenRouter + local Whisper Lite."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import stt_catalog
from stt_context import STTContext
from stt_openrouter import STTProviderError, call_openrouter_transcription

_LOCAL_MODEL = None
_LOCAL_MODEL_KEY: tuple[str, str, str] | None = None
_LOCAL_LOCK_INIT = False


@dataclass
class STTResult:
    text: str
    provider: str
    model: str
    model_id: str = ""
    audio_duration_ms: int | None = None
    transcription_ms: int = 0
    ok: bool = True
    error_type: str | None = None
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    used_fallback: bool = False
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "model_id": self.model_id,
            "audio_duration_ms": self.audio_duration_ms,
            "transcription_ms": self.transcription_ms,
            "ok": self.ok,
            "error_type": self.error_type,
            "request_id": self.request_id,
            "usage": self.usage,
            "cost": self.cost,
            "used_fallback": self.used_fallback,
            "attempts": self.attempts,
        }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def local_whisper_loaded() -> bool:
    return _LOCAL_MODEL is not None


def ensure_local_whisper(
    *,
    model_size: str = "base.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Any:
    """Lazy-load faster-whisper so OpenRouter-primary hosts skip startup cost."""
    global _LOCAL_MODEL, _LOCAL_MODEL_KEY, _LOCAL_LOCK_INIT
    key = (model_size, device, compute_type)
    if _LOCAL_MODEL is not None and _LOCAL_MODEL_KEY == key:
        return _LOCAL_MODEL
    from faster_whisper import WhisperModel

    print(
        f"[STT] loading local whisper model={model_size} device={device} "
        f"compute_type={compute_type}",
        flush=True,
    )
    _LOCAL_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    _LOCAL_MODEL_KEY = key
    _LOCAL_LOCK_INIT = True
    return _LOCAL_MODEL


def _transcribe_local(
    audio_path: Path,
    *,
    entry: dict[str, Any],
    pad_fn: Callable[[Path], Path] | None = None,
    duration_fn: Callable[[Path], float] | None = None,
    peak_fn: Callable[[Path], int | None] | None = None,
) -> tuple[str, int]:
    model = ensure_local_whisper(
        model_size=str(entry.get("model") or "base.en"),
        device=str(entry.get("device") or "cpu"),
        compute_type=str(entry.get("compute_type") or "int8"),
    )
    started = time.perf_counter()
    stt_path = pad_fn(audio_path) if pad_fn else audio_path
    raw_duration = duration_fn(audio_path) if duration_fn else 0.0

    def _run(*, vad_filter: bool, beam_size: int, no_speech_threshold: float) -> str:
        segments, _info = model.transcribe(
            str(stt_path),
            language="en",
            beam_size=beam_size,
            best_of=max(1, beam_size),
            vad_filter=vad_filter,
            condition_on_previous_text=False,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=2.4,
        )
        parts = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    prefer_no_vad = (raw_duration or 0) < 3.0
    if prefer_no_vad:
        attempts = [
            dict(vad_filter=False, beam_size=5, no_speech_threshold=0.35),
            dict(vad_filter=False, beam_size=5, no_speech_threshold=0.2),
            dict(vad_filter=True, beam_size=5, no_speech_threshold=0.35),
        ]
    else:
        attempts = [
            dict(vad_filter=True, beam_size=1, no_speech_threshold=0.6),
            dict(vad_filter=False, beam_size=5, no_speech_threshold=0.35),
            dict(vad_filter=False, beam_size=5, no_speech_threshold=0.2),
        ]

    text = ""
    for attempt in attempts:
        text = _run(**attempt)
        if text:
            break
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(
        f"[STT] request complete provider=local model={entry.get('model')} "
        f"transcription_ms={elapsed_ms} characters={len(text)}",
        flush=True,
    )
    return text, elapsed_ms


def _attempt_openrouter(
    audio_path: Path,
    *,
    entry: dict[str, Any],
    context: STTContext | None,
) -> STTResult:
    model = str(entry.get("model") or "")
    max_retries = max(0, _env_int("STT_MAX_RETRIES", 0))
    timeout = _env_float("STT_TIMEOUT_SECONDS", 15.0)
    enable_keyterms = _env_bool("STT_ENABLE_KEYTERMS", True)
    enable_context = _env_bool("STT_ENABLE_CONTEXT", True)
    language = (os.environ.get("STT_LANGUAGE") or "en").strip() or "en"

    last_error: STTProviderError | None = None
    for attempt_idx in range(max_retries + 1):
        try:
            raw = call_openrouter_transcription(
                audio_path,
                model=model,
                context=context,
                timeout_seconds=timeout,
                enable_keyterms=enable_keyterms,
                enable_context=enable_context,
                language=language,
            )
            if not raw.text:
                raise STTProviderError(
                    "Empty transcript from OpenRouter",
                    error_type="empty_transcript",
                    retryable=False,
                    fallbackable=True,
                )
            return STTResult(
                text=raw.text,
                provider="openrouter",
                model=raw.model,
                model_id=str(entry.get("id") or ""),
                audio_duration_ms=raw.audio_duration_ms,
                transcription_ms=raw.transcription_ms,
                ok=True,
                request_id=raw.request_id,
                usage=raw.usage,
                cost=raw.cost,
            )
        except STTProviderError as ex:
            last_error = ex
            if ex.retryable and attempt_idx < max_retries:
                print(
                    f"[STT] retrying openrouter model={model} "
                    f"attempt={attempt_idx + 2}/{max_retries + 1} error={ex.error_type}",
                    flush=True,
                )
                continue
            raise
    assert last_error is not None
    raise last_error


def transcribe(
    audio_path: Path | str,
    *,
    context: STTContext | None = None,
    pad_fn: Callable[[Path], Path] | None = None,
    duration_fn: Callable[[Path], float] | None = None,
    peak_fn: Callable[[Path], int | None] | None = None,
    hallucination_fn: Callable[..., bool] | None = None,
) -> STTResult:
    """Transcribe audio using the configured STT attempt hierarchy."""
    path = Path(audio_path)
    audio_bytes = path.stat().st_size if path.exists() else 0
    audio_duration_ms = None
    if duration_fn:
        try:
            audio_duration_ms = int(float(duration_fn(path)) * 1000)
        except Exception:
            audio_duration_ms = None

    chain = stt_catalog.get_attempt_chain()
    if not chain:
        return STTResult(
            text="",
            provider="none",
            model="",
            ok=False,
            error_type="unsupported_model",
            audio_duration_ms=audio_duration_ms,
        )

    print(
        f"[STT] hierarchy start chain={chain} audio_bytes={audio_bytes} "
        f"audio_duration_ms={audio_duration_ms}",
        flush=True,
    )

    attempts: list[dict[str, Any]] = []
    last_failure: STTResult | None = None
    primary_id = chain[0]

    for index, entry_id in enumerate(chain):
        entry = stt_catalog.get_entry(entry_id)
        if not entry:
            continue
        provider = str(entry.get("provider") or "")
        model = str(entry.get("model") or "")
        print(
            f"[STT] attempt index={index} model_id={entry_id} provider={provider} "
            f"model={model}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            if provider == "local":
                text, transcription_ms = _transcribe_local(
                    path,
                    entry=entry,
                    pad_fn=pad_fn,
                    duration_fn=duration_fn,
                    peak_fn=peak_fn,
                )
                result = STTResult(
                    text=text,
                    provider="local",
                    model=model,
                    model_id=entry_id,
                    audio_duration_ms=audio_duration_ms,
                    transcription_ms=transcription_ms,
                    ok=bool(text),
                    error_type=None if text else "empty_transcript",
                    used_fallback=entry_id != primary_id,
                )
            else:
                result = _attempt_openrouter(path, entry=entry, context=context)
                result.used_fallback = entry_id != primary_id
                if result.audio_duration_ms is None:
                    result.audio_duration_ms = audio_duration_ms

            if hallucination_fn and result.text:
                peak = peak_fn(path) if peak_fn else None
                duration_s = (result.audio_duration_ms or 0) / 1000.0
                if hallucination_fn(
                    result.text,
                    duration_seconds=duration_s,
                    peak=peak,
                ):
                    print(
                        f"[STT] hallucination rejected model_id={entry_id} "
                        f"text={result.text[:120]!r}",
                        flush=True,
                    )
                    raise STTProviderError(
                        "Transcript rejected as likely hallucination",
                        error_type="empty_transcript",
                        retryable=False,
                        fallbackable=True,
                    )

            if not result.text:
                raise STTProviderError(
                    "Empty transcript",
                    error_type="empty_transcript",
                    retryable=False,
                    fallbackable=True,
                )

            attempts.append(
                {
                    "model_id": entry_id,
                    "provider": result.provider,
                    "model": result.model,
                    "ok": True,
                    "transcription_ms": result.transcription_ms,
                }
            )
            result.attempts = attempts
            return result
        except STTProviderError as ex:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            attempts.append(
                {
                    "model_id": entry_id,
                    "provider": provider,
                    "model": model,
                    "ok": False,
                    "error_type": ex.error_type,
                    "transcription_ms": elapsed_ms,
                }
            )
            print(
                f"[STT] attempt failed model_id={entry_id} error_type={ex.error_type} "
                f"fallbackable={ex.fallbackable} msg={ex}",
                flush=True,
            )
            last_failure = STTResult(
                text="",
                provider=provider or "unknown",
                model=model,
                model_id=entry_id,
                audio_duration_ms=audio_duration_ms,
                transcription_ms=elapsed_ms,
                ok=False,
                error_type=ex.error_type,
                used_fallback=entry_id != primary_id,
                attempts=list(attempts),
            )
            if not ex.fallbackable:
                return last_failure
            continue
        except Exception as ex:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            attempts.append(
                {
                    "model_id": entry_id,
                    "provider": provider,
                    "model": model,
                    "ok": False,
                    "error_type": "local_fallback_failure"
                    if provider == "local"
                    else "provider_error",
                    "transcription_ms": elapsed_ms,
                }
            )
            print(
                f"[STT] attempt exception model_id={entry_id} error={ex!r}",
                flush=True,
            )
            last_failure = STTResult(
                text="",
                provider=provider or "unknown",
                model=model,
                model_id=entry_id,
                audio_duration_ms=audio_duration_ms,
                transcription_ms=elapsed_ms,
                ok=False,
                error_type="local_fallback_failure"
                if provider == "local"
                else "provider_error",
                used_fallback=entry_id != primary_id,
                attempts=list(attempts),
            )
            continue

    if last_failure is not None:
        last_failure.attempts = attempts
        return last_failure
    return STTResult(
        text="",
        provider="none",
        model="",
        ok=False,
        error_type="empty_transcript",
        audio_duration_ms=audio_duration_ms,
        attempts=attempts,
    )


def init_stt(*, base_dir: Path) -> None:
    """Load catalog/runtime from disk (call once at server startup)."""
    stt_catalog.configure(
        catalog_file=base_dir / "stt_catalog.json",
        runtime_file=base_dir / "stt_runtime_config.json",
    )
    stt_catalog.load_or_seed()
    # Eager-load local whisper only when local is primary or mode is local-only.
    chain = stt_catalog.get_attempt_chain()
    mode = stt_catalog.PROVIDER_MODE
    if mode == stt_catalog.PROVIDER_MODE_LOCAL or (
        chain and chain[0] == stt_catalog.LOCAL_WHISPER_ID
    ):
        entry = stt_catalog.get_entry(stt_catalog.LOCAL_WHISPER_ID) or {}
        ensure_local_whisper(
            model_size=str(entry.get("model") or "base.en"),
            device=str(entry.get("device") or "cpu"),
            compute_type=str(entry.get("compute_type") or "int8"),
        )
