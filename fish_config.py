"""Fish speech settings. Read once per call from the process environment.

``FISH_TTS_PROVIDER`` and ``FISH_TTS_MODEL`` select the speech API. They are
not a Piper switch. Invalid values raise ``FishConfigError``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ALLOWED_PROVIDERS = ("fish-audio", "openrouter")
DEFAULT_PROVIDER = "fish-audio"
DEFAULT_MODEL = "fish-audio/drama-3"


class FishConfigError(RuntimeError):
    """FISH_TTS_PROVIDER or FISH_TTS_MODEL cannot be used."""


@dataclass(frozen=True)
class FishSettings:
    provider: str
    model: str
    api_key: str
    openrouter_key: str
    openrouter_base_url: str
    response_format: str
    default_voice_id: str
    timeout_seconds: float
    max_concurrent: int
    catalog_cache_seconds: float
    licensed_only: bool
    progressive: bool
    usd_per_million_utf8: float | None


def _truthy(value: str | None, *, default: bool) -> bool:
    raw = (value if value is not None else "").strip().lower()
    if raw == "":
        return default
    return raw not in ("0", "false", "no", "off")


def _float_env(env: dict[str, str], key: str, default: float) -> float:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    return float(raw)


def native_model_header(model: str) -> str:
    """Fish's HTTP ``model`` header is the slug without the ``fish-audio/`` prefix."""
    prefix = "fish-audio/"
    text = (model or "").strip()
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def load_settings(env: dict[str, str] | None = None) -> FishSettings:
    source = env if env is not None else os.environ
    provider = (source.get("FISH_TTS_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if not provider:
        provider = DEFAULT_PROVIDER
    model = (source.get("FISH_TTS_MODEL") or DEFAULT_MODEL).strip()
    if provider not in ALLOWED_PROVIDERS:
        allowed = ", ".join(ALLOWED_PROVIDERS)
        raise FishConfigError(
            f"FISH_TTS_PROVIDER={provider!r} is invalid. Allowed: {allowed}."
        )
    if not model:
        raise FishConfigError("FISH_TTS_MODEL is empty.")

    price_raw = (source.get("FISH_TTS_USD_PER_MILLION_UTF8") or "").strip()
    price = float(price_raw) if price_raw else None
    return FishSettings(
        provider=provider,
        model=model,
        api_key=(source.get("FISH_API_KEY") or "").strip(),
        openrouter_key=(source.get("OPENROUTER_API_KEY") or "").strip(),
        openrouter_base_url=(
            source.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        response_format=(source.get("FISH_TTS_RESPONSE_FORMAT") or "pcm").strip().lower() or "pcm",
        default_voice_id=(source.get("FISH_DEFAULT_VOICE_ID") or "").strip(),
        timeout_seconds=_float_env(source, "FISH_HTTP_TIMEOUT_SECONDS", 60.0),
        max_concurrent=max(1, int(_float_env(source, "FISH_MAX_CONCURRENT_STREAMS", 4))),
        catalog_cache_seconds=_float_env(source, "FISH_CATALOG_CACHE_SECONDS", 300.0),
        licensed_only=_truthy(source.get("FISH_CATALOG_LICENSED_ONLY"), default=True),
        progressive=_truthy(source.get("FISH_TTS_PROGRESSIVE"), default=False),
        usd_per_million_utf8=price,
    )


def enforce_startup(env: dict[str, str] | None = None) -> FishSettings:
    """Refuse to serve when speech settings or the required key are missing.

    Tests set ``FISH_TTS_SKIP_STARTUP_CHECK=1`` so importing ``server`` does not
    exit. Production does not set that variable.
    """
    source = env if env is not None else os.environ
    if _truthy(source.get("FISH_TTS_SKIP_STARTUP_CHECK"), default=False):
        return load_settings(source)
    settings = load_settings(source)
    if not settings.api_key:
        raise SystemExit("FISH_API_KEY is required for the voice catalog and fish-audio speech.")
    if settings.provider == "openrouter" and not settings.openrouter_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is required when FISH_TTS_PROVIDER=openrouter."
        )
    return settings
