"""
In-memory turn progress tracking for player-facing status updates on the Pi.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

_PROGRESS_LOCK = threading.Lock()
_PROGRESS: dict[str, dict[str, Any]] = {}

REQUEST_ID_RE = re.compile(r"^[a-f0-9]{32}$")
PROGRESS_TTL_SECONDS = 60 * 30

VOICE_TURN_STEPS: tuple[dict[str, Any], ...] = (
    {"key": "registered", "message": "Sending your message to the game server..."},
    {"key": "received", "message": "Message received — getting started..."},
    {"key": "transcribing", "message": "Listening to what you said..."},
    {"key": "loading_context", "message": "Gathering scene, party, and character details..."},
    {"key": "preparing_prompt", "message": "Building the NPC's view of the conversation..."},
    {"key": "waiting_llm", "message": "Waiting for the NPC to think of a reply..."},
    {"key": "processing_response", "message": "Shaping the NPC's response..."},
    {"key": "text_ready", "message": "Reply text ready — preparing voices..."},
    {"key": "generating_voice", "message": "Generating the narrator and NPC voices..."},
    {"key": "clip_ready", "message": "Playing the first response parts..."},
    {"key": "narration_ready", "message": "Narration ready — still generating speech..."},
    {"key": "speech_ready", "message": "Speech ready — finishing up..."},
    {"key": "finishing", "message": "Almost ready — finishing up..."},
    {"key": "complete", "message": "Response ready."},
)

TEXT_TURN_STEPS: tuple[dict[str, Any], ...] = (
    {"key": "registered", "message": "Sending your message to the game server..."},
    {"key": "received", "message": "Message received — getting started..."},
    {"key": "loading_context", "message": "Gathering scene, party, and character details..."},
    {"key": "preparing_prompt", "message": "Building the NPC's view of the conversation..."},
    {"key": "waiting_llm", "message": "Waiting for the NPC to think of a reply..."},
    {"key": "processing_response", "message": "Shaping the NPC's response..."},
    {"key": "text_ready", "message": "Reply text ready..."},
    {"key": "synthesizing_speech", "message": "Synthesizing speech..."},
    {"key": "finishing", "message": "Almost ready — finishing up..."},
    {"key": "complete", "message": "Response ready."},
)

STEP_LOOKUP = {
    "voice": {step["key"]: step for step in VOICE_TURN_STEPS},
    "text": {step["key"]: step for step in TEXT_TURN_STEPS},
}

STEP_ORDER = {
    "voice": [step["key"] for step in VOICE_TURN_STEPS],
    "text": [step["key"] for step in TEXT_TURN_STEPS],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _cleanup_expired() -> None:
    cutoff = time.time() - PROGRESS_TTL_SECONDS
    expired = [
        request_id
        for request_id, entry in _PROGRESS.items()
        if float(entry.get("expires_at") or 0) < cutoff
    ]
    for request_id in expired:
        _PROGRESS.pop(request_id, None)


def normalize_client_request_id(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower().replace("-", "")
    if REQUEST_ID_RE.fullmatch(text):
        return text

    return None


def new_request_id(client_value: Any = None) -> str:
    normalized = normalize_client_request_id(client_value)
    if normalized:
        return normalized
    return uuid.uuid4().hex


def _step_index(mode: str, step_key: str) -> int:
    order = STEP_ORDER.get(mode) or STEP_ORDER["text"]
    try:
        return order.index(step_key) + 1
    except ValueError:
        return 0


def _step_count(mode: str) -> int:
    return len(STEP_ORDER.get(mode) or STEP_ORDER["text"])


def register_turn_progress(
    request_id: str,
    *,
    route: str,
    npc_id: str = "",
    player_id: str = "",
    mode: str = "voice",
) -> dict[str, Any]:
    request_id = new_request_id(request_id)
    mode = "voice" if mode == "voice" else "text"
    now = time.time()

    entry = {
        "request_id": request_id,
        "route": route,
        "npc_id": npc_id,
        "player_id": player_id,
        "mode": mode,
        "status": "in_progress",
        "step": "registered",
        "message": STEP_LOOKUP[mode]["registered"]["message"],
        "step_index": _step_index(mode, "registered"),
        "step_count": _step_count(mode),
        "detail": "",
        "error": "",
        "player_text": "",
        "narration_text": "",
        "dialogue_text": "",
        "clips": [],
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "expires_at": now + PROGRESS_TTL_SECONDS,
    }

    with _PROGRESS_LOCK:
        _cleanup_expired()
        _PROGRESS[request_id] = entry

    return public_progress(entry)


def set_turn_progress(
    request_id: str,
    step_key: str,
    *,
    detail: str = "",
    message: str | None = None,
    player_text: str | None = None,
    narration_text: str | None = None,
    dialogue_text: str | None = None,
    clips: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    request_id = normalize_client_request_id(request_id)
    if not request_id:
        return None

    with _PROGRESS_LOCK:
        entry = _PROGRESS.get(request_id)
        if not entry:
            return None

        # Terminal turns must stay terminal. Re-opening after complete caused player UI
        # to keep polling "Synthesizing speech..." forever after audio already played.
        if entry.get("status") in ("complete", "error", "failed"):
            return public_progress(entry)

        mode = entry.get("mode") or "text"
        step = STEP_LOOKUP.get(mode, STEP_LOOKUP["text"]).get(step_key)
        entry["step"] = step_key
        entry["message"] = message or (step["message"] if step else step_key.replace("_", " ").title())
        entry["detail"] = (detail or "").strip()
        entry["step_index"] = _step_index(mode, step_key)
        entry["step_count"] = _step_count(mode)
        entry["status"] = "in_progress"
        if player_text is not None:
            entry["player_text"] = player_text
        if narration_text is not None:
            entry["narration_text"] = narration_text
        if dialogue_text is not None:
            entry["dialogue_text"] = dialogue_text
        if clips is not None:
            entry["clips"] = list(clips)
        entry["updated_at"] = _utc_now_iso()
        entry["expires_at"] = time.time() + PROGRESS_TTL_SECONDS
        return public_progress(entry)


def complete_turn_progress(request_id: str) -> dict[str, Any] | None:
    return _finish_turn_progress(request_id, status="complete", step_key="complete")


def fail_turn_progress(request_id: str, error: str) -> dict[str, Any] | None:
    return _finish_turn_progress(
        request_id,
        status="error",
        step_key="error",
        error=error,
    )


def _finish_turn_progress(
    request_id: str,
    *,
    status: str,
    step_key: str,
    error: str = "",
) -> dict[str, Any] | None:
    request_id = normalize_client_request_id(request_id)
    if not request_id:
        return None

    with _PROGRESS_LOCK:
        entry = _PROGRESS.get(request_id)
        if not entry:
            return None

        # Terminal turns must stay terminal. A background TTS stream must not
        # flip an HTTP-path fail_turn_progress("error") back to "complete".
        if entry.get("status") in ("complete", "error", "failed"):
            return public_progress(entry)

        mode = entry.get("mode") or "text"
        if status == "error":
            entry["step"] = step_key
            entry["message"] = "Something went wrong while preparing the response."
            entry["error"] = (error or "Unknown error").strip()
        else:
            step = STEP_LOOKUP.get(mode, STEP_LOOKUP["text"]).get(step_key)
            entry["step"] = step_key
            entry["message"] = step["message"] if step else "Response ready."
            entry["error"] = ""

        entry["status"] = status
        entry["step_index"] = _step_count(mode)
        entry["updated_at"] = _utc_now_iso()
        entry["expires_at"] = time.time() + PROGRESS_TTL_SECONDS
        return public_progress(entry)


def get_turn_progress(request_id: str) -> dict[str, Any] | None:
    request_id = normalize_client_request_id(request_id)
    if not request_id:
        return None

    with _PROGRESS_LOCK:
        _cleanup_expired()
        entry = _PROGRESS.get(request_id)
        if not entry:
            return None
        return public_progress(entry)


def public_progress(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": entry.get("request_id"),
        "route": entry.get("route"),
        "npc_id": entry.get("npc_id"),
        "player_id": entry.get("player_id"),
        "mode": entry.get("mode"),
        "status": entry.get("status"),
        "step": entry.get("step"),
        "message": entry.get("message"),
        "detail": entry.get("detail") or "",
        "step_index": entry.get("step_index"),
        "step_count": entry.get("step_count"),
        "error": entry.get("error") or "",
        "player_text": entry.get("player_text") or "",
        "narration_text": entry.get("narration_text") or "",
        "dialogue_text": entry.get("dialogue_text") or "",
        "clips": list(entry.get("clips") or []),
        "created_at": entry.get("created_at"),
        "updated_at": entry.get("updated_at"),
    }


def get_turn_progress_catalog() -> dict[str, Any]:
    return {
        "voice_steps": list(VOICE_TURN_STEPS),
        "text_steps": list(TEXT_TURN_STEPS),
        "endpoints": {
            "register": "/turn/progress/register",
            "status": "/turn/progress/<request_id>",
            "catalog": "/turn/progress/steps",
            "turn_audio": "/turn-audio/<request_id>/<clip>",
        },
        "usage": (
            "Generate a 32-character hex request_id on the Pi, optionally POST "
            "/turn/progress/register before upload, include request_id in the turn "
            "request, and poll GET /turn/progress/<request_id> while waiting. "
            "Narration/speech WAVs are served from /turn-audio/<request_id>/<clip> "
            "once progress reports narration_ready / speech_ready."
        ),
    }
