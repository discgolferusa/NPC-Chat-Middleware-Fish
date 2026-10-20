"""Server-side Fish voice library search. Never downloads the full catalog."""

from __future__ import annotations

import time
from typing import Any, Callable

import requests

from fish_config import FishSettings, load_settings

CATALOG_URL = "https://api.fish.audio/model"
LICENSING_NOTE = (
    "licensed means Fish Audio secured rights from the voice owner. "
    "It is not a commercial-use grant for this product. Public visibility is not enough."
)

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
HttpGet = Callable[..., requests.Response]


def clear_cache() -> None:
    _CACHE.clear()


def _cache_get(key: str, ttl: float) -> dict[str, Any] | None:
    hit = _CACHE.get(key)
    if not hit:
        return None
    stored, payload = hit
    if ttl <= 0 or (time.monotonic() - stored) > ttl:
        return None
    return payload


def _cache_put(key: str, payload: dict[str, Any]) -> None:
    _CACHE[key] = (time.monotonic(), payload)


def _sample_url(item: dict[str, Any]) -> str:
    samples = item.get("samples") or []
    if not isinstance(samples, list):
        return ""
    for sample in samples:
        if isinstance(sample, dict):
            audio = str(sample.get("audio") or "").strip()
            if audio:
                return audio
    return ""


_ACCENT_TAGS = {
    "american",
    "british",
    "australian",
    "irish",
    "scottish",
    "indian",
    "canadian",
    "southern",
    "welsh",
    "african",
    "european",
    "transatlantic",
    "cockney",
    "midwest",
    "new york",
    "english rp",
}


def _language_names(item: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for lang in item.get("languages") or []:
        if isinstance(lang, dict):
            text = str(lang.get("language") or lang.get("code") or "").strip()
        else:
            text = str(lang).strip()
        if text and text not in names:
            names.append(text)
    return names


def _accent_name(item: dict[str, Any], tags: list[str]) -> str:
    explicit = str(item.get("accent") or "").strip()
    if explicit:
        return explicit
    for tag in tags:
        lowered = tag.lower()
        if lowered in _ACCENT_TAGS or "accent" in lowered:
            return tag
    return ""


def to_voice_card(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if str(item.get("type") or "tts") not in ("", "tts"):
        return None
    if item.get("dmca_taken_down"):
        return None
    state = str(item.get("state") or "trained")
    if state not in ("trained", ""):
        return None
    voice_id = str(item.get("_id") or item.get("id") or "").strip()
    title = str(item.get("title") or "").strip()
    if not voice_id or not title:
        return None
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    tags = [str(tag) for tag in (item.get("tags") or []) if str(tag).strip()]
    return {
        "id": voice_id,
        "title": title,
        "description": str(item.get("description") or ""),
        "tags": tags,
        "languages": _language_names(item),
        "accent": _accent_name(item, tags),
        "licensed": bool(item.get("licensed")),
        "preview_url": _sample_url(item),
        "author": str(author.get("nickname") or ""),
        "visibility": str(item.get("visibility") or ""),
        "state": state,
    }


def search_voices(
    *,
    title: str = "",
    tag: str = "",
    language: str = "en",
    page: int = 1,
    page_size: int = 20,
    settings: FishSettings | None = None,
    http_get: HttpGet | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    if not settings.api_key:
        raise RuntimeError("FISH_API_KEY is required to search the Fish voice library.")
    page = max(1, int(page))
    page_size = min(100, max(1, int(page_size)))
    params = {
        "self": "false",
        "page_number": str(page),
        "page_size": str(page_size),
        "sort_by": "score",
    }
    if title.strip():
        params["title"] = title.strip()
    if tag.strip():
        params["tag"] = tag.strip()
    if language.strip():
        params["language"] = language.strip()
    if settings.licensed_only:
        params["licensed"] = "true"
    cache_key = "&".join(f"{key}={params[key]}" for key in sorted(params))
    cached = _cache_get(cache_key, settings.catalog_cache_seconds)
    if cached is not None:
        return cached

    getter = http_get or requests.get
    response = getter(
        CATALOG_URL,
        params=params,
        headers={"Authorization": f"Bearer {settings.api_key}"},
        timeout=settings.timeout_seconds,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Fish voice catalog failed with HTTP {response.status_code}")
    body = response.json()
    cards = []
    for item in body.get("items") or []:
        card = to_voice_card(item)
        if card is not None:
            cards.append(card)
    payload = {
        "provider": "fish-audio",
        "speech_provider": settings.provider,
        "model": settings.model,
        "voices": cards,
        "page": page,
        "page_size": page_size,
        "total": body.get("total"),
        "has_more": body.get("has_more"),
        "licensing_note": LICENSING_NOTE,
        "cached": False,
    }
    _cache_put(cache_key, payload)
    return payload


def get_voice(
    voice_id: str,
    *,
    settings: FishSettings | None = None,
    http_get: HttpGet | None = None,
) -> dict[str, Any] | None:
    """Look up one id without scanning the library. Uses the list title filter only as a fallback.

    Fish's list API does not document a get-by-id query. Callers that already
    selected a voice pass the id; we request ``GET /model/{id}`` and, on 404,
    return None so the turn fails closed instead of substituting another voice.
    """
    settings = settings or load_settings()
    voice_id = (voice_id or "").strip()
    if not voice_id:
        return None
    if not settings.api_key:
        raise RuntimeError("FISH_API_KEY is required to resolve a Fish voice.")
    getter = http_get or requests.get
    response = getter(
        f"{CATALOG_URL}/{voice_id}",
        headers={"Authorization": f"Bearer {settings.api_key}"},
        timeout=settings.timeout_seconds,
    )
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(f"Fish voice lookup failed with HTTP {response.status_code}")
    return to_voice_card(response.json())
