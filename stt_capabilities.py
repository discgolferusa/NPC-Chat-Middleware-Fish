"""STT model capability registry for OpenRouter (and local) transcription models.

Keep model-specific option differences here so route/service code stays generic.
Unknown OpenRouter model slugs get a conservative default (language only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class STTModelCapabilities:
    supports_language: bool = True
    supports_keyterms: bool = False
    supports_context_prompt: bool = False
    max_keyterms: int = 100
    # OpenRouter provider.options slug (assemblyai, deepgram, openai, …).
    provider_options_slug: str | None = None
    # Field name under provider.options.<slug> for free-text context/prompt.
    context_option_key: str | None = None
    # Field name under provider.options.<slug> for keyword lists (Deepgram-style).
    keywords_option_key: str | None = None
    # Prefer top-level OpenRouter ``keyterms`` when True.
    use_top_level_keyterms: bool = False


_DEFAULT = STTModelCapabilities()

# Capabilities keyed by OpenRouter model slug (or local sentinel).
_CAPABILITIES: dict[str, STTModelCapabilities] = {
    "assemblyai/universal-3-5-pro": STTModelCapabilities(
        supports_keyterms=True,
        supports_context_prompt=True,
        max_keyterms=100,
        provider_options_slug="assemblyai",
        context_option_key="prompt",
        use_top_level_keyterms=True,
    ),
    "deepgram/nova-3": STTModelCapabilities(
        supports_keyterms=True,
        supports_context_prompt=False,
        max_keyterms=100,
        provider_options_slug="deepgram",
        keywords_option_key="keywords",
        use_top_level_keyterms=False,
    ),
    "openai/whisper-large-v3-turbo": STTModelCapabilities(
        supports_keyterms=False,
        supports_context_prompt=False,
    ),
    "openai/whisper-large-v3": STTModelCapabilities(
        supports_keyterms=False,
        supports_context_prompt=False,
    ),
    "openai/gpt-transcribe": STTModelCapabilities(
        supports_keyterms=True,
        supports_context_prompt=True,
        max_keyterms=100,
        provider_options_slug="openai",
        context_option_key="prompt",
        use_top_level_keyterms=True,
    ),
    "openai/gpt-4o-transcribe": STTModelCapabilities(
        supports_keyterms=True,
        supports_context_prompt=True,
        max_keyterms=100,
        provider_options_slug="openai",
        context_option_key="prompt",
        use_top_level_keyterms=True,
    ),
    # Local faster-whisper — no cloud keyterms/prompt.
    "base.en": STTModelCapabilities(supports_keyterms=False, supports_context_prompt=False),
    "tiny.en": STTModelCapabilities(supports_keyterms=False, supports_context_prompt=False),
    "small.en": STTModelCapabilities(supports_keyterms=False, supports_context_prompt=False),
}


def get_capabilities(model: str) -> STTModelCapabilities:
    key = (model or "").strip()
    if key in _CAPABILITIES:
        return _CAPABILITIES[key]
    # Infer lightly from vendor prefix without scattering ifs in callers.
    vendor = key.split("/", 1)[0].casefold() if "/" in key else ""
    if vendor == "assemblyai":
        return STTModelCapabilities(
            supports_keyterms=True,
            supports_context_prompt=True,
            max_keyterms=100,
            provider_options_slug="assemblyai",
            context_option_key="prompt",
            use_top_level_keyterms=True,
        )
    if vendor == "deepgram":
        return STTModelCapabilities(
            supports_keyterms=True,
            supports_context_prompt=False,
            max_keyterms=100,
            provider_options_slug="deepgram",
            keywords_option_key="keywords",
            use_top_level_keyterms=False,
        )
    if vendor == "openai" and "whisper" not in key.casefold():
        return STTModelCapabilities(
            supports_keyterms=True,
            supports_context_prompt=True,
            max_keyterms=100,
            provider_options_slug="openai",
            context_option_key="prompt",
            use_top_level_keyterms=True,
        )
    return _DEFAULT


def capabilities_public_dict(model: str) -> dict[str, Any]:
    caps = get_capabilities(model)
    return {
        "supports_language": caps.supports_language,
        "supports_keyterms": caps.supports_keyterms,
        "supports_context_prompt": caps.supports_context_prompt,
        "max_keyterms": caps.max_keyterms,
    }


def register_capabilities(model: str, caps: STTModelCapabilities) -> None:
    """Test/helper hook to override or extend the registry at runtime."""
    _CAPABILITIES[(model or "").strip()] = caps
