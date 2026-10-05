"""OpenRouter speech-to-text HTTP client (``/audio/transcriptions``)."""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from stt_capabilities import get_capabilities
from stt_context import STTContext, apply_context_to_request


class STTProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_type: str = "provider_error",
        status_code: int | None = None,
        retryable: bool = False,
        fallbackable: bool = True,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable
        self.fallbackable = fallbackable


@dataclass
class OpenRouterSTTRawResult:
    text: str
    model: str
    transcription_ms: int
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    audio_duration_ms: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _audio_format_from_path(audio_path: Path) -> str:
    suffix = (audio_path.suffix or "").lstrip(".").lower()
    if suffix in {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}:
        return suffix
    return "wav"


def call_openrouter_transcription(
    audio_path: Path,
    *,
    model: str,
    context: STTContext | None = None,
    timeout_seconds: float | None = None,
    enable_keyterms: bool = True,
    enable_context: bool = True,
    language: str | None = None,
) -> OpenRouterSTTRawResult:
    """POST base64 audio to OpenRouter transcriptions endpoint."""
    import llm_router

    api_key = (llm_router.OPENROUTER_API_KEY or "").strip()
    if not api_key:
        raise STTProviderError(
            "OPENROUTER_API_KEY is not configured",
            error_type="authentication",
            status_code=401,
            retryable=False,
            fallbackable=True,
        )

    audio_bytes = audio_path.read_bytes()
    if not audio_bytes:
        raise STTProviderError(
            "Audio file is empty",
            error_type="malformed_audio",
            status_code=400,
            retryable=False,
            fallbackable=False,
        )

    fmt = _audio_format_from_path(audio_path)
    caps = get_capabilities(model)
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else os.environ.get("STT_TIMEOUT_SECONDS", "15")
    )

    payload: dict[str, Any] = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": fmt,
        },
    }
    effective_context = context
    if language and (context is None or not context.language):
        effective_context = STTContext(
            keyterms=list(context.keyterms) if context else [],
            prompt=(context.prompt if context else ""),
            language=language,
        )
    apply_context_to_request(
        payload,
        effective_context,
        caps,
        enable_keyterms=enable_keyterms,
        enable_context=enable_context,
    )

    url = f"{llm_router._openrouter_api_root().rstrip('/')}/audio/transcriptions"
    headers = llm_router._openrouter_auth_headers()

    print(
        f"[STT] request started provider=openrouter model={model} "
        f"audio_bytes={len(audio_bytes)} format={fmt} timeout={timeout}s "
        f"keyterms={len(payload.get('keyterms') or [])}",
        flush=True,
    )

    started = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout as ex:
        raise STTProviderError(
            f"OpenRouter STT timed out after {timeout}s",
            error_type="timeout",
            retryable=True,
            fallbackable=True,
        ) from ex
    except requests.RequestException as ex:
        raise STTProviderError(
            f"OpenRouter unavailable: {ex}",
            error_type="openrouter_unavailable",
            retryable=True,
            fallbackable=True,
        ) from ex

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    request_id = response.headers.get("X-Generation-Id") or response.headers.get(
        "x-generation-id"
    )

    if response.status_code in (401, 403):
        raise STTProviderError(
            f"OpenRouter authentication failed ({response.status_code})",
            error_type="authentication",
            status_code=response.status_code,
            retryable=False,
            fallbackable=True,
        )
    if response.status_code == 429:
        raise STTProviderError(
            "OpenRouter rate limited",
            error_type="rate_limit",
            status_code=429,
            retryable=False,
            fallbackable=True,
        )
    if response.status_code == 402:
        raise STTProviderError(
            "OpenRouter credits exhausted",
            error_type="credits_exhausted",
            status_code=402,
            retryable=False,
            fallbackable=True,
        )
    if response.status_code in (500, 502, 503, 504):
        raise STTProviderError(
            f"OpenRouter upstream error ({response.status_code})",
            error_type="openrouter_unavailable",
            status_code=response.status_code,
            retryable=True,
            fallbackable=True,
        )
    if response.status_code >= 400:
        # Try to distinguish unsupported model vs bad audio.
        body_text = (response.text or "")[:300]
        error_type = "invalid_response"
        fallbackable = True
        lower = body_text.casefold()
        if "keyterm" in lower or "unsupported" in lower or "not supported" in lower:
            error_type = "unsupported_model"
        if "audio" in lower and ("invalid" in lower or "malformed" in lower):
            error_type = "malformed_audio"
            fallbackable = False
        raise STTProviderError(
            f"OpenRouter STT error ({response.status_code}): {body_text}",
            error_type=error_type,
            status_code=response.status_code,
            retryable=False,
            fallbackable=fallbackable,
        )

    try:
        data = response.json()
    except ValueError as ex:
        raise STTProviderError(
            "OpenRouter returned non-JSON STT response",
            error_type="invalid_response",
            status_code=response.status_code,
            retryable=False,
            fallbackable=True,
        ) from ex

    if not isinstance(data, dict):
        raise STTProviderError(
            "OpenRouter STT response was not an object",
            error_type="invalid_response",
            retryable=False,
            fallbackable=True,
        )

    text = str(data.get("text") or "").strip()
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    cost = None
    if usage.get("cost") is not None:
        try:
            cost = float(usage["cost"])
        except (TypeError, ValueError):
            cost = None

    audio_duration_ms = None
    duration = data.get("duration")
    if duration is None and usage:
        duration = usage.get("seconds")
    if duration is not None:
        try:
            audio_duration_ms = int(float(duration) * 1000)
        except (TypeError, ValueError):
            audio_duration_ms = None

    print(
        f"[STT] request complete provider=openrouter model={model} "
        f"transcription_ms={elapsed_ms} characters={len(text)} "
        f"request_id={request_id or '-'}",
        flush=True,
    )

    return OpenRouterSTTRawResult(
        text=text,
        model=model,
        transcription_ms=elapsed_ms,
        request_id=request_id,
        usage=dict(usage),
        cost=cost,
        audio_duration_ms=audio_duration_ms,
        raw=data,
    )
