"""Fish speech client. One provider per process, chosen by environment."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

import requests

from fish_config import FishSettings, load_settings, native_model_header

class FishVoiceError(RuntimeError):
    """The NPC has no Fish voice and no explicit default is configured."""


class FishSpeechError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()
_SEMAPHORE: threading.BoundedSemaphore | None = None
_SEMAPHORE_SIZE = 0
_LAST: dict[str, Any] = {}
_LAST_LOCK = threading.Lock()


def session() -> requests.Session:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = requests.Session()
        return _SESSION


def _semaphore(size: int) -> threading.BoundedSemaphore:
    global _SEMAPHORE, _SEMAPHORE_SIZE
    if _SEMAPHORE is None or _SEMAPHORE_SIZE != size:
        _SEMAPHORE = threading.BoundedSemaphore(size)
        _SEMAPHORE_SIZE = size
    return _SEMAPHORE


def last_call_metrics() -> dict[str, Any]:
    with _LAST_LOCK:
        return dict(_LAST)


def _remember(metrics: dict[str, Any]) -> None:
    with _LAST_LOCK:
        _LAST.clear()
        _LAST.update(metrics)


def reset_client_for_tests() -> None:
    global _SESSION, _SEMAPHORE, _SEMAPHORE_SIZE
    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION.close()
        _SESSION = None
    _SEMAPHORE = None
    _SEMAPHORE_SIZE = 0
    _remember({})


def _parse_content_type(value: str) -> tuple[str, int, int]:
    """Return (subtype, sample_rate, channels). Rate defaults to 44100 mono for raw PCM."""
    text = (value or "").split(";")[0].strip().lower()
    subtype = "pcm"
    if "mpeg" in text or text.endswith("mp3"):
        subtype = "mp3"
    elif "wav" in text:
        subtype = "wav"
    rate = 44100
    channels = 1
    for part in (value or "").split(";")[1:]:
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        key = key.strip().lower()
        raw = raw.strip()
        if key in ("rate", "samplerate", "sample_rate") and raw.isdigit():
            rate = int(raw)
        if key == "channels" and raw.isdigit():
            channels = int(raw)
    return subtype, rate, channels


def pcm_to_wav(pcm: bytes, path: Path, *, sample_rate: int, channels: int) -> None:
    even = pcm[: len(pcm) - (len(pcm) % (2 * max(1, channels)))]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(max(1, channels))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(even)


def _request_parts(text: str, voice_id: str, settings: FishSettings) -> tuple[str, dict[str, str], dict[str, Any]]:
    if settings.provider == "openrouter":
        url = f"{settings.openrouter_base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {settings.openrouter_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": settings.model,
            "input": text,
            "voice": voice_id,
            "response_format": settings.response_format,
        }
        return url, headers, body
    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        "model": native_model_header(settings.model),
    }
    body = {
        "text": text,
        "reference_id": voice_id,
        "format": settings.response_format,
        "latency": "balanced",
    }
    return url, headers, body


def synthesize_wav(
    text: str,
    voice_id: str | None,
    output_path: Path,
    *,
    settings: FishSettings | None = None,
    http_post: Callable[..., requests.Response] | None = None,
    cancel_event: threading.Event | None = None,
    on_pcm_chunk: Callable[[bytes, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Write a WAV file. Does not call Piper and does not retry another provider."""
    settings = settings or load_settings()
    resolved = (voice_id or "").strip() or settings.default_voice_id
    started = time.perf_counter()
    metrics: dict[str, Any] = {
        "provider": settings.provider,
        "model": settings.model,
        "voice_id": resolved,
        "utf8_bytes": len((text or "").encode("utf-8")),
        "estimated_usd": None,
        "ttfab_ms": None,
        "first_decoded_ms": None,
        "error": None,
    }
    if settings.usd_per_million_utf8 is not None:
        metrics["estimated_usd"] = round(
            metrics["utf8_bytes"] / 1_000_000 * settings.usd_per_million_utf8,
            6,
        )
    if not resolved:
        metrics["error"] = "missing_voice"
        _remember(metrics)
        raise FishVoiceError(
            "NPC has no fish_voice_id and FISH_DEFAULT_VOICE_ID is unset."
        )
    if settings.provider == "openrouter" and not settings.openrouter_key:
        metrics["error"] = "missing_openrouter_key"
        _remember(metrics)
        raise FishSpeechError("OPENROUTER_API_KEY is required when FISH_TTS_PROVIDER=openrouter.")
    if settings.provider == "fish-audio" and not settings.api_key:
        metrics["error"] = "missing_fish_key"
        _remember(metrics)
        raise FishSpeechError("FISH_API_KEY is required when FISH_TTS_PROVIDER=fish-audio.")

    url, headers, body = _request_parts(text or "", resolved, settings)
    poster = http_post or session().post
    sem = _semaphore(settings.max_concurrent)
    sem.acquire()
    response: requests.Response | None = None
    try:
        response = poster(url, headers=headers, json=body, stream=True, timeout=settings.timeout_seconds)
        status = int(getattr(response, "status_code", 0) or 0)
        if status >= 400:
            detail = ""
            try:
                detail = response.text[:300]
            except Exception:
                detail = ""
            metrics["error"] = f"http_{status}"
            _remember(metrics)
            response.close()
            response = None
            raise FishSpeechError(
                f"Fish speech failed with HTTP {status}. {detail}".strip(),
                status=status,
            )
        content_type = ""
        if hasattr(response, "headers"):
            content_type = str(response.headers.get("Content-Type") or "")
        subtype, rate, channels = _parse_content_type(content_type or settings.response_format)
        chunks: list[bytes] = []
        first_byte: float | None = None
        seq = 0
        pending = b""
        iterator = response.iter_content(chunk_size=4096) if hasattr(response, "iter_content") else [response.content]
        for piece in iterator:
            if cancel_event is not None and cancel_event.is_set():
                metrics["error"] = "cancelled"
                _remember(metrics)
                raise FishSpeechError("Fish speech cancelled.")
            if not piece:
                continue
            if first_byte is None:
                first_byte = time.perf_counter()
                metrics["ttfab_ms"] = max(0, int(round((first_byte - started) * 1000)))
            chunks.append(piece)
            if subtype == "pcm" and on_pcm_chunk is not None:
                pending += piece
                aligned = len(pending) - (len(pending) % (2 * max(1, channels)))
                if aligned:
                    frame = pending[:aligned]
                    pending = pending[aligned:]
                    seq += 1
                    if metrics["first_decoded_ms"] is None:
                        metrics["first_decoded_ms"] = metrics["ttfab_ms"]
                    on_pcm_chunk(frame, seq, rate, channels)
        audio = b"".join(chunks)
        if not audio:
            metrics["error"] = "empty_audio"
            _remember(metrics)
            raise FishSpeechError("Fish speech returned an empty body.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if subtype == "wav":
            output_path.write_bytes(audio)
        elif subtype == "mp3":
            mp3_path = output_path.with_suffix(".mp3")
            mp3_path.write_bytes(audio)
            _ffmpeg_to_wav(mp3_path, output_path)
            mp3_path.unlink(missing_ok=True)
        else:
            pcm_to_wav(audio, output_path, sample_rate=rate, channels=channels)
            if metrics["first_decoded_ms"] is None:
                metrics["first_decoded_ms"] = metrics["ttfab_ms"]
        metrics["audio_bytes"] = output_path.stat().st_size
        metrics["sample_rate"] = rate
        metrics["channels"] = channels
        metrics["synth_ms"] = max(0, int(round((time.perf_counter() - started) * 1000)))
        _remember(metrics)
        return metrics
    finally:
        if response is not None:
            response.close()
        sem.release()


def _ffmpeg_to_wav(src: Path, dest: Path) -> None:
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "22050", str(dest)],
        check=True,
        capture_output=True,
    )


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate() or 1
        return handle.getnframes() / float(rate)


def read_wav_pcm(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes()), handle.getframerate(), handle.getnchannels()
