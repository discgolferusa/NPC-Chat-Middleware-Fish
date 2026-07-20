"""
Session scene state: time of day and current sub-location for LLM prompts.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent
LOCATION_DIR = BASE_DIR / "locations"
SCENE_RUNTIME_CONFIG_FILE = BASE_DIR / "scene_runtime_config.json"

TIME_OF_DAY_OPTIONS = (
    "Pre-Dawn",
    "Morning",
    "Midday",
    "Afternoon",
    "Evening",
    "Overnight",
)

_STATE_LOCK = threading.Lock()
_LOCATION_CACHE_LOCK = threading.Lock()
LOCATION_CACHE: dict[str, dict[str, Any]] = {}

# session_key(owner, campaign) -> { time_of_day, sub_location, updated_at, touched_at }
_SCENE_SESSIONS: dict[str, dict[str, Any]] = {}
_MAX_SCENE_SESSIONS = 2000

# Local/dev fallback when no owner+campaign context is present.
_FALLBACK_TIME_OF_DAY = ""
_FALLBACK_SUB_LOCATION: dict[str, Any] | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_time_of_day(value: str | None) -> str:
    value = (value or "").strip()
    if not value:
        return ""

    for option in TIME_OF_DAY_OPTIONS:
        if value.casefold() == option.casefold():
            return option

    raise ValueError(
        f"Invalid time_of_day: {value}. "
        f"Expected one of: {', '.join(TIME_OF_DAY_OPTIONS)}"
    )


def _load_location_yaml(location_id: str) -> dict[str, Any]:
    location_id = (location_id or "").strip()
    if not location_id:
        return {}

    try:
        from content_pack import lookup_document

        packed = lookup_document("location", location_id)
        if packed is not None:
            data = deepcopy(packed)
            data.setdefault("id", location_id)
            return data
    except Exception:
        pass

    for suffix in (".yaml", ".yml"):
        path = LOCATION_DIR / f"{location_id}{suffix}"
        if not path.exists():
            continue

        modified_at = path.stat().st_mtime
        with _LOCATION_CACHE_LOCK:
            cached = LOCATION_CACHE.get(location_id)
            if cached and cached.get("modified_at") == modified_at:
                return deepcopy(cached["data"])

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        data.setdefault("id", location_id)
        with _LOCATION_CACHE_LOCK:
            LOCATION_CACHE[location_id] = {"modified_at": modified_at, "data": data}
        return deepcopy(data)

    return {
        "id": location_id,
        "name": location_id,
        "error": f"Location file not found for location_id: {location_id}",
    }


def _scene_bucket() -> dict[str, Any]:
    from session_context import session_key

    key = session_key()
    if not key:
        return {
            "time_of_day": _FALLBACK_TIME_OF_DAY,
            "sub_location": deepcopy(_FALLBACK_SUB_LOCATION),
            "updated_at": None,
            "touched_at": 0.0,
            "_fallback": True,
        }

    with _STATE_LOCK:
        bucket = _SCENE_SESSIONS.get(key)
        if bucket is None:
            bucket = {
                "time_of_day": "",
                "sub_location": None,
                "updated_at": None,
                "touched_at": datetime.now(timezone.utc).timestamp(),
            }
            _SCENE_SESSIONS[key] = bucket
            if len(_SCENE_SESSIONS) > _MAX_SCENE_SESSIONS:
                ordered = sorted(
                    _SCENE_SESSIONS.items(),
                    key=lambda item: float(item[1].get("touched_at") or 0),
                )
                for old_key, _ in ordered[: len(_SCENE_SESSIONS) - _MAX_SCENE_SESSIONS]:
                    _SCENE_SESSIONS.pop(old_key, None)
        else:
            bucket["touched_at"] = datetime.now(timezone.utc).timestamp()
        return bucket


def _load_runtime_config() -> None:
    global _FALLBACK_TIME_OF_DAY, _FALLBACK_SUB_LOCATION

    if not SCENE_RUNTIME_CONFIG_FILE.exists():
        _FALLBACK_TIME_OF_DAY = ""
        _FALLBACK_SUB_LOCATION = None
        return

    try:
        data = json.loads(SCENE_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _FALLBACK_TIME_OF_DAY = ""
        _FALLBACK_SUB_LOCATION = None
        return

    try:
        _FALLBACK_TIME_OF_DAY = _normalize_time_of_day(data.get("time_of_day"))
    except ValueError:
        _FALLBACK_TIME_OF_DAY = ""

    sub_location = data.get("sub_location")
    _FALLBACK_SUB_LOCATION = deepcopy(sub_location) if isinstance(sub_location, dict) else None


def _save_runtime_config() -> None:
    # Persist only anonymous/local fallback; concurrent tables stay in memory.
    bucket = _scene_bucket()
    payload = {
        "time_of_day": bucket.get("time_of_day") or "",
        "sub_location": bucket.get("sub_location"),
        "updated_at": _utc_now_iso(),
        "note": "Fallback defaults for local/dev; live scene state is keyed by owner+campaign in memory.",
    }
    try:
        SCENE_RUNTIME_CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as ex:
        print(f"[SCENE] failed to save runtime config: {ex}", flush=True)


def _find_important_location(location: dict[str, Any], name: str) -> dict[str, Any] | None:
    target = (name or "").strip().casefold()
    if not target:
        return None

    for item in location.get("important_locations") or []:
        if not isinstance(item, dict):
            continue
        item_name = (item.get("name") or "").strip()
        if item_name.casefold() == target:
            return item

    return None


def _normalize_sub_location(
    *,
    location_id: str,
    important_location_name: str | None = None,
    custom: str | None = None,
) -> dict[str, Any]:
    location_id = (location_id or "").strip()
    if not location_id:
        raise ValueError("Missing location_id")

    important_location_name = (important_location_name or "").strip()
    custom = (custom or "").strip()

    if important_location_name and custom:
        raise ValueError("Provide either important_location_name or custom, not both")

    if not important_location_name and not custom:
        raise ValueError("Provide important_location_name or custom")

    location = _load_location_yaml(location_id)
    if location.get("error"):
        raise ValueError(location["error"])

    if important_location_name:
        match = _find_important_location(location, important_location_name)
        if not match:
            known = [
                (item.get("name") or "").strip()
                for item in (location.get("important_locations") or [])
                if isinstance(item, dict) and (item.get("name") or "").strip()
            ]
            known_text = ", ".join(known) if known else "(none listed)"
            raise ValueError(
                f"Unknown important_location_name: {important_location_name}. "
                f"Known places in {location_id}: {known_text}"
            )

        return {
            "location_id": location_id,
            "location_name": location.get("name") or location_id,
            "name": match.get("name") or important_location_name,
            "description": (match.get("description") or "").strip(),
            "public_knowledge": (match.get("public_knowledge") or "").strip(),
            "source": "important_location",
        }

    return {
        "location_id": location_id,
        "location_name": location.get("name") or location_id,
        "name": custom,
        "description": "",
        "public_knowledge": "",
        "source": "custom",
    }


def set_time_of_day(time_of_day: str) -> dict[str, Any]:
    normalized = _normalize_time_of_day(time_of_day)
    if not normalized:
        raise ValueError("Missing time_of_day")

    bucket = _scene_bucket()
    if bucket.get("_fallback"):
        global _FALLBACK_TIME_OF_DAY
        _FALLBACK_TIME_OF_DAY = normalized
    else:
        with _STATE_LOCK:
            bucket["time_of_day"] = normalized
            bucket["updated_at"] = _utc_now_iso()
    _save_runtime_config()
    return get_scene_status()


def clear_time_of_day() -> dict[str, Any]:
    bucket = _scene_bucket()
    if bucket.get("_fallback"):
        global _FALLBACK_TIME_OF_DAY
        _FALLBACK_TIME_OF_DAY = ""
    else:
        with _STATE_LOCK:
            bucket["time_of_day"] = ""
            bucket["updated_at"] = _utc_now_iso()
    _save_runtime_config()
    return get_scene_status()


def set_sub_location(
    *,
    location_id: str,
    important_location_name: str | None = None,
    custom: str | None = None,
) -> dict[str, Any]:
    sub_location = _normalize_sub_location(
        location_id=location_id,
        important_location_name=important_location_name,
        custom=custom,
    )

    bucket = _scene_bucket()
    if bucket.get("_fallback"):
        global _FALLBACK_SUB_LOCATION
        _FALLBACK_SUB_LOCATION = sub_location
    else:
        with _STATE_LOCK:
            bucket["sub_location"] = sub_location
            bucket["updated_at"] = _utc_now_iso()
    _save_runtime_config()
    return get_scene_status()


def clear_sub_location() -> dict[str, Any]:
    bucket = _scene_bucket()
    if bucket.get("_fallback"):
        global _FALLBACK_SUB_LOCATION
        _FALLBACK_SUB_LOCATION = None
    else:
        with _STATE_LOCK:
            bucket["sub_location"] = None
            bucket["updated_at"] = _utc_now_iso()
    _save_runtime_config()
    return get_scene_status()


def list_sub_location_options(location_id: str) -> dict[str, Any]:
    location_id = (location_id or "").strip()
    if not location_id:
        raise ValueError("Missing location_id")

    location = _load_location_yaml(location_id)
    if location.get("error"):
        raise ValueError(location["error"])

    important_locations = []
    for item in location.get("important_locations") or []:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if not name:
            continue
        important_locations.append({
            "name": name,
            "description": (item.get("description") or "").strip(),
            "public_knowledge": (item.get("public_knowledge") or "").strip(),
        })

    return {
        "location_id": location_id,
        "location_name": location.get("name") or location_id,
        "important_locations": important_locations,
    }


def build_scene_context(npc_location_id: str | None = None) -> dict[str, Any]:
    npc_location_id = (npc_location_id or "").strip()
    bucket = _scene_bucket()
    time_of_day = bucket.get("time_of_day") or ""
    sub_location = deepcopy(bucket.get("sub_location"))

    applies = bool(
        sub_location
        and npc_location_id
        and (sub_location.get("location_id") or "").strip() == npc_location_id
    )

    return {
        "time_of_day": time_of_day,
        "sub_location": sub_location,
        "npc_location_id": npc_location_id,
        "sub_location_applies": applies,
    }


def format_scene_context_for_prompt(scene_context: dict[str, Any] | None) -> str:
    scene_context = scene_context or {}
    lines: list[str] = []

    time_of_day = (scene_context.get("time_of_day") or "").strip()
    if time_of_day:
        lines.append(f"Current time of day: {time_of_day} (authoritative; do not invent a different time).")
    else:
        lines.append(
            "Current time of day: not set by the DM. Do not assume night, morning, or any specific time "
            "unless the players establish it in conversation."
        )

    sub_location = scene_context.get("sub_location")
    applies = scene_context.get("sub_location_applies")
    npc_location_id = (scene_context.get("npc_location_id") or "").strip()

    if sub_location and applies:
        place_name = (sub_location.get("name") or "").strip()
        location_name = (sub_location.get("location_name") or sub_location.get("location_id") or "").strip()
        description = (sub_location.get("description") or "").strip()

        lines.append(
            f"Current scene place: {place_name} in {location_name} "
            "(authoritative; this is where the conversation is happening right now)."
        )
        if description:
            lines.append(f"Scene place details: {description}")
    elif sub_location and npc_location_id and not applies:
        lines.append(
            "Current scene place: not set for this NPC's region. Do not assume a specific building, "
            "street, or landmark unless the players establish it in conversation."
        )
    else:
        region_hint = f" Region context: {npc_location_id}." if npc_location_id else ""
        lines.append(
            "Current scene place: not set by the DM. Do not assume the party is at the Old Aqueduct, "
            "a specific inn, or any other landmark unless established in conversation."
            f"{region_hint}"
        )

    lines.append(
        "Scene guidance: use the stated time and place as background truth. Do not repeatedly mention "
        "the time of day or location in every reply. Do not relocate the scene or change the time of day "
        "on your own. Regional knowledge below describes the wider area, not necessarily where this "
        "conversation is happening."
    )

    return "\n".join(lines)


def get_scene_status(*, include_options: bool = True) -> dict[str, Any]:
    bucket = _scene_bucket()
    time_of_day = bucket.get("time_of_day") or ""
    sub_location = deepcopy(bucket.get("sub_location"))

    payload: dict[str, Any] = {
        "time_of_day": time_of_day or None,
        "sub_location": sub_location,
        "updated_at": bucket.get("updated_at"),
    }

    if include_options:
        payload["time_of_day_options"] = list(TIME_OF_DAY_OPTIONS)
        payload["endpoints"] = {
            "scene_status": "/scene/status",
            "scene_time": "/scene/time",
            "scene_time_clear": "/scene/time/clear",
            "scene_sub_location": "/scene/sub-location",
            "scene_sub_location_clear": "/scene/sub-location/clear",
            "scene_sub_location_options": "/scene/sub-locations",
        }

    return payload


def init_scene_state() -> None:
    LOCATION_DIR.mkdir(exist_ok=True)
    _load_runtime_config()
    print(
        f"[SCENE] multi-table ready "
        f"(fallback time={_FALLBACK_TIME_OF_DAY or '(unset)'} "
        f"place={(_FALLBACK_SUB_LOCATION or {}).get('name') or '(unset)'})",
        flush=True,
    )


init_scene_state()
