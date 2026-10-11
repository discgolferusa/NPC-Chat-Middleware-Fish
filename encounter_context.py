"""Shared encounter lines for one NPC, with an audience snapshot per line.

Legacy rows (NULL visibility) stay visible only to the character who spoke them.
Scene lines are audible to the audience stored on the row. Private lines, including
Discord and Foundry whispers, stay with the speaking character. The NPC still
receives private dialogue in its own knowledge block and is told not to disclose it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import memory_db

ENCOUNTER_DIALOGUE_LIMIT = 12
ENCOUNTER_SUMMARY_LIMIT = 8
ENCOUNTER_RESOLUTION_LIMIT = 12
ENCOUNTER_FETCH_LIMIT = 80

VISIBILITY_SCENE = "scene"
VISIBILITY_PRIVATE = "private"
EVENT_DIALOGUE = "dialogue"
EVENT_RESOLUTION = "resolution"


def _clean_ids(values: Any) -> list[str]:
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in text.split(",")]
        values = parsed
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        player_id = str(item or "").strip()
        if not player_id or player_id in seen:
            continue
        seen.add(player_id)
        cleaned.append(player_id)
    return cleaned


def dump_audience(player_ids: list[str] | None) -> str | None:
    cleaned = _clean_ids(player_ids)
    if not cleaned:
        return None
    return json.dumps(cleaned)


def present_player_ids(party_context: dict[str, Any] | None) -> list[str]:
    """Party members who are at the scene. Empty absent list means everyone."""
    from scene_state import get_absent_player_ids

    absent = set(get_absent_player_ids())
    present: list[str] = []
    for member in (party_context or {}).get("party_members") or []:
        player_id = str(member.get("player_id") or "").strip()
        if player_id and player_id not in absent and player_id not in present:
            present.append(player_id)
    speaker = str((party_context or {}).get("player_id") or "").strip()
    if speaker and speaker not in present and speaker not in absent:
        present.append(speaker)
    return present


def resolve_turn_visibility(
    *,
    visibility: str | None,
    explicit_audience: Any = None,
    speaker_id: str,
    present_ids: list[str] | None,
) -> tuple[str, list[str]]:
    """Map a request onto scene or private audience.

    Discord/Foundry ``whisper`` and an explicit audience are player-only (private).
    Anything else is scene-audible to whoever is currently present.
    """
    speaker = (speaker_id or "").strip()
    raw = (visibility or "").strip().lower()
    if raw in ("whisper", "private"):
        return VISIBILITY_PRIVATE, [speaker] if speaker else []

    explicit = _clean_ids(explicit_audience)
    if explicit:
        if speaker and speaker not in explicit:
            explicit = [speaker, *explicit]
        return VISIBILITY_PRIVATE, explicit

    present = _clean_ids(present_ids)
    if speaker and speaker not in present:
        present = [*present, speaker]
    if not present and speaker:
        present = [speaker]
    return VISIBILITY_SCENE, present


def format_scene_ref(scene_context: dict[str, Any] | None) -> str:
    scene_context = scene_context or {}
    parts: list[str] = []
    sub = scene_context.get("sub_location") if scene_context.get("sub_location_applies") else None
    if isinstance(sub, dict):
        place = (sub.get("name") or "").strip()
        location = (sub.get("location_name") or sub.get("location_id") or "").strip()
        if place and location:
            parts.append(f"{place} in {location}")
        elif place or location:
            parts.append(place or location)
    else:
        region = (scene_context.get("npc_location_id") or "").strip()
        if region:
            parts.append(region)
    time_of_day = (scene_context.get("time_of_day") or "").strip()
    if time_of_day:
        parts.append(time_of_day)
    return " | ".join(parts)


def build_turn_record(
    *,
    visibility: str | None,
    explicit_audience: Any,
    speaker_id: str,
    party_context: dict[str, Any] | None,
    scene_context: dict[str, Any] | None,
    event_kind: str = EVENT_DIALOGUE,
) -> dict[str, Any]:
    vis, audience = resolve_turn_visibility(
        visibility=visibility,
        explicit_audience=explicit_audience,
        speaker_id=speaker_id,
        present_ids=present_player_ids(party_context),
    )
    return {
        "visibility": vis,
        "audience_player_ids": audience,
        "scene_ref": format_scene_ref(scene_context),
        "event_kind": event_kind or EVENT_DIALOGUE,
    }


def _conversation_turns_ready(connection: memory_db.DbConnection) -> bool:
    """DM-only setups initialize rulings without the conversation table."""
    return "conversation_turns" in memory_db._existing_tables(connection)


def insert_turn(
    npc_id: str,
    player_id: str,
    role: str,
    content: str,
    campaign_id: str = "",
    *,
    visibility: str | None = None,
    audience_player_ids: list[str] | None = None,
    scene_ref: str | None = None,
    event_kind: str | None = None,
    connection: memory_db.DbConnection | None = None,
) -> None:
    content = (content or "").strip()
    if not content:
        return
    campaign_id = (campaign_id or "").strip()
    visibility = (visibility or "").strip().lower() or None
    if visibility not in (None, VISIBILITY_SCENE, VISIBILITY_PRIVATE):
        visibility = VISIBILITY_PRIVATE if visibility in ("whisper",) else VISIBILITY_SCENE
    audience_json = dump_audience(audience_player_ids) if visibility else None
    scene_ref = (scene_ref or "").strip() or None
    event_kind = (event_kind or "").strip() or None
    if visibility and not event_kind:
        event_kind = EVENT_DIALOGUE

    sql = """
        INSERT INTO conversation_turns
            (campaign_id, npc_id, player_id, role, content, created_at,
             visibility, audience_player_ids, scene_ref, event_kind)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        campaign_id,
        npc_id,
        player_id,
        role,
        content,
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        visibility,
        audience_json,
        scene_ref,
        event_kind,
    )
    if connection is not None:
        if _conversation_turns_ready(connection):
            connection.execute(sql, params)
        return
    with memory_db.get_db() as conn:
        if _conversation_turns_ready(conn):
            conn.execute(sql, params)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    try:
        keys = row.keys()
    except AttributeError:
        return default
    if key not in keys:
        return default
    value = row[key]
    return default if value is None else value


def _turn_from_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "player_id": row["player_id"],
        "role": row["role"],
        "content": row["content"] or "",
        "visibility": (_row_get(row, "visibility", "") or "").strip().lower(),
        "audience_player_ids": _clean_ids(_row_get(row, "audience_player_ids", "")),
        "scene_ref": (_row_get(row, "scene_ref", "") or "").strip(),
        "event_kind": (_row_get(row, "event_kind", "") or "").strip().lower(),
    }


def load_encounter_turns(
    npc_id: str,
    campaign_id: str = "",
    *,
    connection: memory_db.DbConnection | None = None,
    limit: int = ENCOUNTER_FETCH_LIMIT,
) -> list[dict[str, Any]]:
    """All recent lines for this NPC, oldest first. Caller applies visibility."""
    campaign_id = (campaign_id or "").strip()
    sql = """
        SELECT id, player_id, role, content, visibility, audience_player_ids,
               scene_ref, event_kind
        FROM conversation_turns
        WHERE campaign_id = ?
          AND npc_id = ?
        ORDER BY id DESC
        LIMIT ?
    """
    params = (campaign_id, npc_id, limit)
    if connection is not None:
        rows = connection.execute(sql, params).fetchall()
    else:
        with memory_db.get_db() as conn:
            rows = conn.execute(sql, params).fetchall()
    return [_turn_from_row(row) for row in reversed(rows)]


def turn_visible_to_npc(turn: dict[str, Any], current_player_id: str) -> bool:
    visibility = (turn.get("visibility") or "").strip().lower()
    if not visibility:
        return (turn.get("player_id") or "") == current_player_id
    kind = (turn.get("event_kind") or "").strip().lower()
    if kind == EVENT_RESOLUTION or (turn.get("role") or "") == "gm":
        # A private GM resolution is omitted entirely unless this character heard it.
        if visibility == VISIBILITY_PRIVATE:
            return player_heard_turn(turn, current_player_id)
        return visibility == VISIBILITY_SCENE
    return visibility in (VISIBILITY_SCENE, VISIBILITY_PRIVATE)


def player_heard_turn(turn: dict[str, Any], current_player_id: str) -> bool:
    visibility = (turn.get("visibility") or "").strip().lower()
    if not visibility:
        return (turn.get("player_id") or "") == current_player_id
    if visibility == VISIBILITY_PRIVATE:
        audience = turn.get("audience_player_ids") or []
        return current_player_id in audience or turn.get("player_id") == current_player_id
    if visibility == VISIBILITY_SCENE:
        audience = turn.get("audience_player_ids") or []
        if not audience:
            return True
        return current_player_id in audience
    return False


def _member_names(party_context: dict[str, Any] | None) -> dict[str, str]:
    names: dict[str, str] = {}
    for member in (party_context or {}).get("party_members") or []:
        player_id = str(member.get("player_id") or "").strip()
        if not player_id:
            continue
        names[player_id] = (member.get("name") or player_id).strip() or player_id
    speaking = (party_context or {}).get("speaking_member") or {}
    speaker_id = str(speaking.get("player_id") or "").strip()
    if speaker_id and speaker_id not in names:
        names[speaker_id] = (speaking.get("name") or speaker_id).strip() or speaker_id
    return names


def _display_name(player_id: str, names: dict[str, str]) -> str:
    return names.get(player_id) or player_id or "someone"


def _heard_label(turn: dict[str, Any], names: dict[str, str]) -> str:
    visibility = (turn.get("visibility") or "").strip().lower()
    if not visibility:
        return _display_name(turn.get("player_id") or "", names) + " only (earlier private history)"
    if visibility == VISIBILITY_PRIVATE:
        audience = turn.get("audience_player_ids") or []
        if audience:
            return ", ".join(_display_name(pid, names) for pid in audience) + " only"
        return _display_name(turn.get("player_id") or "", names) + " only"
    audience = turn.get("audience_player_ids") or []
    if not audience:
        return "everyone present"
    return ", ".join(_display_name(pid, names) for pid in audience)


def format_turn_line(turn: dict[str, Any], names: dict[str, str]) -> str:
    actor = _display_name(turn.get("player_id") or "", names)
    heard = _heard_label(turn, names)
    visibility = (turn.get("visibility") or "").strip().lower() or "legacy"
    place = (turn.get("scene_ref") or "").strip()
    place_bit = f" | {place}" if place else ""
    content = (turn.get("content") or "").strip()
    kind = (turn.get("event_kind") or "").strip().lower()
    if kind == EVENT_RESOLUTION or (turn.get("role") or "") == "gm":
        return (
            f"[GM RESOLUTION | actor: {actor} | heard by: {heard} | {visibility}{place_bit}] "
            f"{content}"
        )
    if (turn.get("role") or "") == "assistant":
        speaker = "NPC"
    else:
        speaker = actor
    return f"[{speaker} | heard by: {heard} | {visibility}{place_bit}]: {content}"


def _summary_line(turn: dict[str, Any], names: dict[str, str]) -> str:
    place = (turn.get("scene_ref") or "").strip() or "an earlier moment"
    if (turn.get("role") or "") == "assistant":
        speaker = "the NPC"
    else:
        speaker = _display_name(turn.get("player_id") or "", names)
    text = " ".join((turn.get("content") or "").split())
    if len(text) > 80:
        text = text[:79].rstrip() + "…"
    return f"Earlier at {place}: {speaker} said {text}"


def select_encounter_window(
    turns: list[dict[str, Any]],
    current_player_id: str,
    *,
    dialogue_limit: int = ENCOUNTER_DIALOGUE_LIMIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (older dialogue to summarize, recent dialogue, resolutions to keep)."""
    visible = [turn for turn in turns if turn_visible_to_npc(turn, current_player_id)]
    dialogue = [
        turn for turn in visible
        if (turn.get("event_kind") or EVENT_DIALOGUE) != EVENT_RESOLUTION
        and (turn.get("role") or "") != "gm"
    ]
    resolutions = [
        turn for turn in visible
        if (turn.get("event_kind") or "") == EVENT_RESOLUTION or (turn.get("role") or "") == "gm"
    ]
    if len(dialogue) > dialogue_limit:
        older = dialogue[:-dialogue_limit]
        recent = dialogue[-dialogue_limit:]
    else:
        older = []
        recent = dialogue
    older = older[-ENCOUNTER_SUMMARY_LIMIT:]
    resolutions = resolutions[-ENCOUNTER_RESOLUTION_LIMIT:]
    return older, recent, resolutions


def format_encounter_prompt(
    turns: list[dict[str, Any]] | None,
    *,
    current_player_id: str,
    party_context: dict[str, Any] | None,
) -> str:
    names = _member_names(party_context)
    speaker_name = _display_name(current_player_id, names)
    older, recent, resolutions = select_encounter_window(turns or [], current_player_id)

    know_lines = [_summary_line(turn, names) for turn in older]
    know_lines.extend(format_turn_line(turn, names) for turn in recent)
    know_lines.extend(format_turn_line(turn, names) for turn in resolutions)

    heard_lines = [
        format_turn_line(turn, names)
        for turn in [*recent, *resolutions]
        if player_heard_turn(turn, current_player_id)
    ]

    if not know_lines:
        know_block = "No encounter lines are on record for this NPC yet."
    else:
        know_block = "\n".join(f"- {line}" for line in know_lines)
    if not heard_lines:
        heard_block = f"{speaker_name} has not heard any of these lines."
    else:
        heard_block = "\n".join(f"- {line}" for line in heard_lines)

    return (
        f"The character now speaking is {speaker_name} ({current_player_id}).\n"
        "What you know because you took part in it (do not treat another NPC as knowing this):\n"
        f"{know_block}\n\n"
        f"What {speaker_name} heard:\n"
        f"{heard_block}\n\n"
        f"Do not reveal lines that {speaker_name} did not hear, including private or whisper lines. "
        "Do not ask again for an answer already given by someone who is still listed as having heard it. "
        "A promise, debt, or item stays with the named actor. "
        "Something a character reports is not something you saw. "
        "A GM RESOLUTION authorizes only the recorded consequence."
    )


def attribute_memory(memory: str, actor_name: str) -> str:
    text = (memory or "").strip()
    name = (actor_name or "").strip()
    if not text or not name:
        return text
    if name.casefold() in text.casefold():
        return text
    return f"{name}: {text}"


def load_encounter_memories(
    npc_id: str,
    player_id: str,
    campaign_id: str = "",
    *,
    limit: int = 8,
    connection: memory_db.DbConnection | None = None,
) -> list[str]:
    """Scene memories for this NPC from any character; private and legacy stay player-only."""
    campaign_id = (campaign_id or "").strip()
    sql = """
        SELECT player_id, memory, visibility
        FROM npc_memories
        WHERE campaign_id = ?
          AND npc_id = ?
        ORDER BY id DESC
        LIMIT ?
    """
    params = (campaign_id, npc_id, max(limit * 4, 20))
    if connection is not None:
        rows = connection.execute(sql, params).fetchall()
    else:
        with memory_db.get_db() as conn:
            rows = conn.execute(sql, params).fetchall()

    kept: list[str] = []
    for row in reversed(rows):
        visibility = (_row_get(row, "visibility", "") or "").strip().lower()
        owner = row["player_id"]
        if visibility == VISIBILITY_SCENE:
            kept.append(row["memory"])
        elif owner == player_id:
            kept.append(row["memory"])
    return kept[-limit:]


def latest_turn_scope(
    npc_id: str,
    player_id: str,
    campaign_id: str = "",
    *,
    connection: memory_db.DbConnection | None = None,
) -> tuple[str, list[str]]:
    """Visibility of the latest line from this character to this NPC.

    Returns ("", []) when that line is legacy or missing. Private is preserved.
    """
    campaign_id = (campaign_id or "").strip()
    sql = """
        SELECT visibility, audience_player_ids
        FROM conversation_turns
        WHERE campaign_id = ?
          AND npc_id = ?
          AND player_id = ?
        ORDER BY id DESC
        LIMIT 1
    """
    params = (campaign_id, npc_id, player_id)
    if connection is not None:
        if not _conversation_turns_ready(connection):
            return "", []
        row = connection.execute(sql, params).fetchone()
    else:
        with memory_db.get_db() as conn:
            if not _conversation_turns_ready(conn):
                return "", []
            row = conn.execute(sql, params).fetchone()
    if not row:
        return "", []
    visibility = (_row_get(row, "visibility", "") or "").strip().lower()
    if visibility == VISIBILITY_PRIVATE:
        audience = _clean_ids(_row_get(row, "audience_player_ids", ""))
        if player_id and player_id not in audience:
            audience = [player_id, *audience]
        return VISIBILITY_PRIVATE, audience or [player_id]
    return "", []
