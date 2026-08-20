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
# One in-flight turn per (campaign_id, npc_id, player_id) — blocks overlapping player/DM replies.
_ACTIVE_TURNS: dict[tuple[str, str, str], str] = {}


class TurnInProgressError(Exception):
    """Raised when a second turn starts while the same speaker already has one in flight."""

    def __init__(self, progress: dict[str, Any]):
        self.progress = progress
        super().__init__(progress.get("message") or "turn_in_progress")

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


def _normalize_campaign_id(campaign_id: Any) -> str:
    return str(campaign_id or "").strip().lower()


def _session_key(campaign_id: str, npc_id: str, player_id: str) -> tuple[str, str, str] | None:
    npc = (npc_id or "").strip()
    player = (player_id or "").strip()
    if not npc or not player:
        return None
    return (_normalize_campaign_id(campaign_id), npc, player)


def _release_active_turn(entry: dict[str, Any]) -> None:
    key = _session_key(
        entry.get("campaign_id") or "",
        entry.get("npc_id") or "",
        entry.get("player_id") or "",
    )
    if not key:
        return
    if _ACTIVE_TURNS.get(key) == entry.get("request_id"):
        _ACTIVE_TURNS.pop(key, None)


def _cleanup_expired() -> None:
    cutoff = time.time() - PROGRESS_TTL_SECONDS
    expired = [
        request_id
        for request_id, entry in _PROGRESS.items()
        if float(entry.get("expires_at") or 0) < cutoff
    ]
    for request_id in expired:
        entry = _PROGRESS.pop(request_id, None)
        if entry:
            _release_active_turn(entry)


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


def get_active_turn_progress(
    campaign_id: str,
    npc_id: str,
    player_id: str,
) -> dict[str, Any] | None:
    key = _session_key(campaign_id, npc_id, player_id)
    if not key:
        return None

    with _PROGRESS_LOCK:
        _cleanup_expired()
        request_id = _ACTIVE_TURNS.get(key)
        if not request_id:
            return None
        entry = _PROGRESS.get(request_id)
        if not entry or entry.get("status") != "in_progress":
            _ACTIVE_TURNS.pop(key, None)
            return None
        return public_progress(entry)


def register_turn_progress(
    request_id: str,
    *,
    route: str,
    npc_id: str = "",
    player_id: str = "",
    mode: str = "voice",
    campaign_id: str = "",
    force: bool = False,
) -> dict[str, Any]:
    request_id = new_request_id(request_id)
    mode = "voice" if mode == "voice" else "text"
    now = time.time()

    entry = {
        "request_id": request_id,
        "route": route,
        "npc_id": npc_id,
        "player_id": player_id,
        "campaign_id": _normalize_campaign_id(campaign_id),
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
        key = _session_key(entry["campaign_id"], npc_id, player_id)
        if key:
            existing_id = _ACTIVE_TURNS.get(key)
            if existing_id and existing_id != request_id:
                existing = _PROGRESS.get(existing_id)
                if existing and existing.get("status") == "in_progress":
                    if not force:
                        raise TurnInProgressError(public_progress(existing))
                    _release_active_turn(existing)
            _ACTIVE_TURNS[key] = request_id
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
        if status in ("complete", "error", "failed"):
            _release_active_turn(entry)
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
        "campaign_id": entry.get("campaign_id") or "",
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
            "active": "/turn/progress/active",
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
