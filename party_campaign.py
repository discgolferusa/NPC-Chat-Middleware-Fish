"""
Party and campaign YAML loading, session state, and prompt formatting.
"""

from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parent
PARTIES_DIR = BASE_DIR / "parties"
CAMPAIGNS_DIR = BASE_DIR / "campaigns"
SESSION_RUNTIME_CONFIG_FILE = BASE_DIR / "session_runtime_config.json"

DEFAULT_PLAYER_ID = os.getenv("DEFAULT_PLAYER_ID", "default_party")
DEFAULT_PARTY_ID = os.getenv("DEFAULT_PARTY_ID", "default_party")
DEFAULT_CAMPAIGN_ID = os.getenv("DEFAULT_CAMPAIGN_ID", "")

DEFAULT_PARTY_MEMBER: dict[str, Any] = {
    "player_id": DEFAULT_PLAYER_ID,
    "party_id": DEFAULT_PARTY_ID,
    "campaign_id": "",
    "name": "The Party",
    "race": "Unknown",
    "class": "Adventurer",
    "label": "Default Party",
    "is_default": True,
}

_STATE_LOCK = threading.Lock()
_YAML_CACHE_LOCK = threading.Lock()
PARTY_CACHE: dict[str, dict[str, Any]] = {}
CAMPAIGN_CACHE: dict[str, dict[str, Any]] = {}

# (owner_user_id, campaign_id) -> play session (party/player selection for that table)
_PLAY_SESSIONS: dict[str, dict[str, Any]] = {}
_MAX_PLAY_SESSIONS = 2000

# Legacy single-process defaults only used when request has no owner/campaign context
# (local smoke tests without headers).
_FALLBACK_CAMPAIGN_ID = DEFAULT_CAMPAIGN_ID
_FALLBACK_PARTY_ID = DEFAULT_PARTY_ID
_FALLBACK_PLAYER_ID = DEFAULT_PLAYER_ID


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def init_party_campaign_dirs() -> None:
    PARTIES_DIR.mkdir(exist_ok=True)
    CAMPAIGNS_DIR.mkdir(exist_ok=True)


def _load_yaml_cached(path: Path, cache: dict[str, dict[str, Any]], cache_key: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")

    modified_at = path.stat().st_mtime
    with _YAML_CACHE_LOCK:
        cached = cache.get(cache_key)
        if cached and cached.get("modified_at") == modified_at:
            return deepcopy(cached["data"])

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    with _YAML_CACHE_LOCK:
        cache[cache_key] = {"modified_at": modified_at, "data": data}

    return deepcopy(data)


def _party_file_paths() -> list[Path]:
    paths = list(PARTIES_DIR.glob("*.yaml")) + list(PARTIES_DIR.glob("*.yml"))
    return sorted(paths, key=lambda path: path.stem.lower())


def _campaign_file_paths() -> list[Path]:
    paths = list(CAMPAIGNS_DIR.glob("*.yaml")) + list(CAMPAIGNS_DIR.glob("*.yml"))
    return sorted(paths, key=lambda path: path.stem.lower())


def load_campaign(campaign_id: str) -> dict[str, Any]:
    campaign_id = (campaign_id or "").strip()
    if not campaign_id:
        return {}

    from content_pack import lookup_document

    packed = lookup_document("campaign", campaign_id)
    if packed is not None:
        data = deepcopy(packed)
        data.setdefault("campaign_id", campaign_id)
        data.setdefault("id", campaign_id)
        return data

    for path in _campaign_file_paths():
        if path.stem.casefold() == campaign_id.casefold():
            data = _load_yaml_cached(path, CAMPAIGN_CACHE, campaign_id)
            data.setdefault("campaign_id", campaign_id)
            data.setdefault("id", campaign_id)
            return data

    # Filename may differ from campaign_id (editor lists by YAML campaign_id).
    for path in _campaign_file_paths():
        data = _load_yaml_cached(path, CAMPAIGN_CACHE, path.stem)
        embedded = str(data.get("campaign_id") or data.get("id") or "").strip()
        if embedded and embedded.casefold() == campaign_id.casefold():
            data.setdefault("campaign_id", campaign_id)
            data.setdefault("id", campaign_id)
            return data

    return {
        "campaign_id": campaign_id,
        "id": campaign_id,
        "name": campaign_id,
        "description": "",
        "error": f"Campaign file not found for campaign_id: {campaign_id}",
    }


def load_party(party_id: str) -> dict[str, Any]:
    party_id = (party_id or "").strip()
    if not party_id:
        return {}

    from content_pack import lookup_document

    packed = lookup_document("party", party_id)
    if packed is not None:
        data = deepcopy(packed)
        data.setdefault("party_id", party_id)
        data.setdefault("id", party_id)
        return normalize_party_data(data)

    for path in _party_file_paths():
        if path.stem == party_id:
            data = _load_yaml_cached(path, PARTY_CACHE, party_id)
            data.setdefault("party_id", party_id)
            data.setdefault("id", party_id)
            return normalize_party_data(data)

    return {
        "party_id": party_id,
        "id": party_id,
        "label": party_id,
        "campaign_id": "",
        "members": [],
        "error": f"Party file not found for party_id: {party_id}",
    }


def normalize_party_data(party: dict[str, Any]) -> dict[str, Any]:
    party = deepcopy(party)
    party_id = party.get("party_id") or party.get("id") or "default_party"
    campaign_id = party.get("campaign_id") or ""
    party["party_id"] = party_id
    party["id"] = party_id
    party["campaign_id"] = campaign_id

    members = []
    for raw_member in party.get("members") or []:
        if not isinstance(raw_member, dict):
            continue

        member = dict(raw_member)
        member["player_id"] = (member.get("player_id") or "").strip()
        if not member["player_id"]:
            continue

        member["party_id"] = member.get("party_id") or party_id
        member["campaign_id"] = member.get("campaign_id") or campaign_id
        member["name"] = member.get("name") or member["player_id"]
        member["race"] = member.get("race") or "Unknown"
        member["class"] = member.get("class") or "Adventurer"
        member["is_default"] = False
        members.append(member)

    party["members"] = members
    return party


def list_campaigns() -> list[dict[str, Any]]:
    from content_pack import has_pack, list_document_ids

    campaigns = []
    seen: set[str] = set()

    if has_pack():
        for campaign_id in list_document_ids("campaign"):
            seen.add(campaign_id.casefold())
            try:
                data = load_campaign(campaign_id)
                campaigns.append({
                    "campaign_id": data.get("campaign_id", campaign_id),
                    "id": data.get("campaign_id", campaign_id),
                    "name": data.get("name", campaign_id),
                    "description": data.get("description", ""),
                })
            except Exception as ex:
                campaigns.append({
                    "campaign_id": campaign_id,
                    "id": campaign_id,
                    "name": campaign_id,
                    "description": "",
                    "error": str(ex),
                })
        return campaigns

    for path in _campaign_file_paths():
        campaign_id = path.stem
        if campaign_id.casefold() in seen:
            continue
        try:
            data = load_campaign(campaign_id)
            campaigns.append({
                "campaign_id": data.get("campaign_id", campaign_id),
                "id": data.get("campaign_id", campaign_id),
                "name": data.get("name", campaign_id),
                "description": data.get("description", ""),
            })
        except Exception as ex:
            campaigns.append({
                "campaign_id": campaign_id,
                "id": campaign_id,
                "name": campaign_id,
                "description": "",
                "error": str(ex),
            })
    return campaigns


def list_parties(campaign_id: str | None = None) -> list[dict[str, Any]]:
    from content_pack import has_pack, list_document_ids

    campaign_id = (campaign_id or "").strip()
    parties = []
    party_ids: list[str] = []

    if has_pack():
        party_ids = list_document_ids("party")
    else:
        party_ids = [path.stem for path in _party_file_paths()]

    for party_id in party_ids:
        try:
            party = load_party(party_id)
            if campaign_id and party.get("campaign_id") and party.get("campaign_id") != campaign_id:
                continue

            parties.append({
                "party_id": party.get("party_id", party_id),
                "id": party.get("party_id", party_id),
                "label": party.get("label", party.get("name", party_id)),
                "campaign_id": party.get("campaign_id", ""),
                "member_count": len(party.get("members") or []),
            })
        except Exception as ex:
            parties.append({
                "party_id": party_id,
                "id": party_id,
                "label": party_id,
                "campaign_id": "",
                "member_count": 0,
                "error": str(ex),
            })

    return parties


def find_member(party_id: str, player_id: str) -> dict[str, Any] | None:
    party = load_party(party_id)
    for member in party.get("members") or []:
        if member.get("player_id") == player_id:
            return deepcopy(member)
    return None


def find_member_anywhere(player_id: str) -> dict[str, Any] | None:
    from content_pack import has_pack, list_document_ids

    player_id = (player_id or "").strip()
    if not player_id:
        return None

    party_ids = list_document_ids("party") if has_pack() else [path.stem for path in _party_file_paths()]
    for party_id in party_ids:
        party = load_party(party_id)
        for member in party.get("members") or []:
            if member.get("player_id") == player_id:
                return deepcopy(member)

    return None


def get_default_party_member(campaign_id: str | None = None) -> dict[str, Any]:
    member = deepcopy(DEFAULT_PARTY_MEMBER)
    if campaign_id:
        member["campaign_id"] = campaign_id
    return member


def _character_options_for_parties(party_ids: list[str], campaign_id: str) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for current_party_id in party_ids:
        party = load_party(current_party_id)
        if campaign_id and party.get("campaign_id") and party.get("campaign_id") != campaign_id:
            continue

        for member in party.get("members") or []:
            player_id = member.get("player_id")
            if not player_id or player_id in seen_ids:
                continue
            seen_ids.add(player_id)
            options.append({
                **member,
                "label": f"{member.get('name')} ({member.get('race')} {member.get('class')})",
            })

    return options


def _campaign_party_ids(campaign_id: str) -> list[str]:
    return [item["party_id"] for item in list_parties(campaign_id=campaign_id or None)]


def get_character_options(campaign_id: str | None = None, party_id: str | None = None) -> list[dict[str, Any]]:
    campaign_id = (campaign_id or "").strip()
    party_id = (party_id or "").strip()

    if party_id:
        options = _character_options_for_parties([party_id], campaign_id)
        # A restart resets the play session to the placeholder party id, which
        # resolves to nothing. Widening to the campaign keeps real characters
        # in the list instead of collapsing it to the default party.
        if not options:
            options = _character_options_for_parties(_campaign_party_ids(campaign_id), campaign_id)
    else:
        options = _character_options_for_parties(_campaign_party_ids(campaign_id), campaign_id)

    if not options:
        default_member = get_default_party_member(campaign_id)
        options.append({
            **default_member,
            "label": default_member.get("label", "Default Party"),
        })

    return options


def _utc_now_ts() -> float:
    return time.time()


def _prune_play_sessions_locked() -> None:
    if len(_PLAY_SESSIONS) <= _MAX_PLAY_SESSIONS:
        return
    ordered = sorted(
        _PLAY_SESSIONS.items(),
        key=lambda item: float(item[1].get("touched_at") or 0),
    )
    overflow = len(_PLAY_SESSIONS) - _MAX_PLAY_SESSIONS
    for key, _ in ordered[:overflow]:
        _PLAY_SESSIONS.pop(key, None)


def get_or_create_play_session(
    owner_user_id: str,
    campaign_id: str,
    *,
    party_id: str | None = None,
    player_id: str | None = None,
) -> dict[str, Any]:
    from session_context import set_request_context

    owner_user_id = (owner_user_id or "").strip()
    campaign_id = (campaign_id or "").strip()
    if not owner_user_id or not campaign_id:
        raise ValueError("owner_user_id and campaign_id are required for a play session")

    set_request_context(owner_user_id, campaign_id)
    key = f"{owner_user_id.casefold()}\x1f{campaign_id.casefold()}"
    with _STATE_LOCK:
        session = _PLAY_SESSIONS.get(key)
        if session is None:
            session = {
                "owner_user_id": owner_user_id,
                "campaign_id": campaign_id,
                "party_id": (party_id or DEFAULT_PARTY_ID).strip() or DEFAULT_PARTY_ID,
                "player_id": (player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID,
                "touched_at": _utc_now_ts(),
            }
            _PLAY_SESSIONS[key] = session
            _prune_play_sessions_locked()
        else:
            if party_id:
                session["party_id"] = party_id.strip() or session["party_id"]
            if player_id:
                session["player_id"] = player_id.strip() or session["player_id"]
            session["touched_at"] = _utc_now_ts()
        return dict(session)


def _current_play_ids() -> dict[str, str]:
    from session_context import get_request_campaign, get_request_owner

    owner = get_request_owner()
    campaign = get_request_campaign()
    if owner and campaign:
        session = get_or_create_play_session(owner, campaign)
        return {
            "owner_user_id": session["owner_user_id"],
            "campaign_id": session["campaign_id"],
            "party_id": session.get("party_id") or DEFAULT_PARTY_ID,
            "player_id": session.get("player_id") or DEFAULT_PLAYER_ID,
        }

    return {
        "owner_user_id": owner,
        "campaign_id": campaign or _FALLBACK_CAMPAIGN_ID or "",
        "party_id": _FALLBACK_PARTY_ID,
        "player_id": _FALLBACK_PLAYER_ID,
    }


def _load_runtime_config() -> None:
    """Load legacy single-table file into fallbacks for local/dev only."""
    global _FALLBACK_CAMPAIGN_ID, _FALLBACK_PARTY_ID, _FALLBACK_PLAYER_ID

    if not SESSION_RUNTIME_CONFIG_FILE.exists():
        return

    try:
        data = json.loads(SESSION_RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    _FALLBACK_CAMPAIGN_ID = (data.get("campaign_id") or DEFAULT_CAMPAIGN_ID or "").strip()
    _FALLBACK_PARTY_ID = (data.get("party_id") or DEFAULT_PARTY_ID).strip()
    _FALLBACK_PLAYER_ID = (data.get("player_id") or DEFAULT_PLAYER_ID).strip()


def _save_runtime_config() -> None:
    # Persist only the anonymous/local fallback — concurrent tables stay in memory.
    ids = _current_play_ids()
    payload = {
        "campaign_id": ids["campaign_id"],
        "party_id": ids["party_id"],
        "player_id": ids["player_id"],
        "updated_at": _utc_now_iso(),
        "note": "Fallback defaults for local/dev; live tables are keyed by owner+campaign in memory.",
    }
    try:
        SESSION_RUNTIME_CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as ex:
        print(f"[SESSION] failed to save runtime config: {ex}", flush=True)


def select_campaign(campaign_id: str, *, owner_user_id: str | None = None) -> dict[str, Any]:
    from session_context import get_request_owner, set_request_context

    campaign_id = (campaign_id or "").strip()
    if not campaign_id:
        raise ValueError("Missing campaign_id")

    owner = (owner_user_id or get_request_owner() or "").strip()
    if not owner:
        # Local/dev without multi-tenant headers.
        owner = "local"

    set_request_context(owner, campaign_id)

    campaign = load_campaign(campaign_id)
    if campaign.get("error"):
        raise ValueError(campaign["error"])

    current = get_or_create_play_session(owner, campaign_id)
    current_party_id = current.get("party_id") or DEFAULT_PARTY_ID
    current_player_id = current.get("player_id") or DEFAULT_PLAYER_ID

    parties = list_parties(campaign_id=campaign_id)
    if parties:
        new_party_id = (
            current_party_id
            if current_party_id in {party["party_id"] for party in parties}
            else parties[0]["party_id"]
        )
    else:
        new_party_id = DEFAULT_PARTY_ID

    characters = get_character_options(campaign_id=campaign_id, party_id=new_party_id)
    valid_player_ids = {character["player_id"] for character in characters}
    if valid_player_ids:
        new_player_id = (
            current_player_id
            if current_player_id in valid_player_ids
            else characters[0]["player_id"]
        )
    else:
        new_player_id = DEFAULT_PLAYER_ID

    get_or_create_play_session(
        owner,
        campaign_id,
        party_id=new_party_id,
        player_id=new_player_id,
    )
    _save_runtime_config()
    return get_session_status()


def select_character(player_id: str, *, party_id: str | None = None) -> dict[str, Any]:
    from session_context import get_request_campaign, get_request_owner, set_request_context

    player_id = (player_id or "").strip()
    if not player_id:
        raise ValueError("Missing player_id")

    owner = get_request_owner() or "local"
    current = _current_play_ids()
    current_party_id = current["party_id"]
    current_campaign_id = current["campaign_id"] or get_request_campaign()

    new_party_id = party_id.strip() if party_id else current_party_id
    new_campaign_id = current_campaign_id

    characters = get_character_options(
        campaign_id=new_campaign_id or None,
        party_id=new_party_id or None,
    )
    valid_ids = {character["player_id"] for character in characters}
    if player_id not in valid_ids:
        member = find_member_anywhere(player_id)
        if member:
            new_party_id = member.get("party_id") or new_party_id
            if member.get("campaign_id"):
                new_campaign_id = member["campaign_id"]
        elif player_id != DEFAULT_PLAYER_ID:
            raise ValueError(f"Unknown player_id: {player_id}")

    if not new_campaign_id:
        raise ValueError("campaign_id is required to select a character")

    set_request_context(owner, new_campaign_id)
    get_or_create_play_session(
        owner,
        new_campaign_id,
        party_id=new_party_id,
        player_id=player_id,
    )
    _save_runtime_config()
    return get_session_status()


def resolve_member(player_id: str | None = None, party_id: str | None = None) -> dict[str, Any]:
    ids = _current_play_ids()
    player_id = (player_id or ids["player_id"] or DEFAULT_PLAYER_ID).strip()
    party_id = (party_id or ids["party_id"] or DEFAULT_PARTY_ID).strip()

    member = find_member(party_id, player_id)
    if member:
        return member

    member = find_member_anywhere(player_id)
    if member:
        return member

    if player_id == DEFAULT_PLAYER_ID:
        return get_default_party_member(ids["campaign_id"] or None)

    return {
        "player_id": player_id,
        "party_id": party_id,
        "campaign_id": ids["campaign_id"] or "",
        "name": player_id,
        "race": "Unknown",
        "class": "Adventurer",
        "is_default": False,
    }


def build_party_context(player_id: str | None = None, party_id: str | None = None) -> dict[str, Any]:
    ids = _current_play_ids()
    player_id = (player_id or ids["player_id"] or DEFAULT_PLAYER_ID).strip()
    party_id = (party_id or ids["party_id"] or DEFAULT_PARTY_ID).strip()
    campaign_id = (ids["campaign_id"] or "").strip()

    speaking_member = resolve_member(player_id, party_id)
    party_id = speaking_member.get("party_id") or party_id
    campaign_id = speaking_member.get("campaign_id") or campaign_id or ids["campaign_id"]

    party = load_party(party_id) if party_id else {"members": []}
    if party.get("error"):
        party = {"party_id": party_id, "members": [], "campaign_id": campaign_id}

    members = party.get("members") or []
    if not members and speaking_member.get("is_default"):
        members = [speaking_member]

    campaign = load_campaign(campaign_id) if campaign_id else {}

    party_members = []
    for member in members:
        party_members.append({
            **member,
            "is_speaking": member.get("player_id") == speaking_member.get("player_id"),
        })

    if speaking_member.get("player_id") == DEFAULT_PLAYER_ID and not any(
        m.get("player_id") == DEFAULT_PLAYER_ID for m in party_members
    ):
        party_members.insert(0, {**speaking_member, "is_speaking": True})

    return {
        "campaign_id": campaign_id,
        "campaign": campaign,
        "party_id": party_id,
        "party_label": party.get("label") or party.get("name") or party_id,
        "player_id": speaking_member.get("player_id"),
        "speaking_member": speaking_member,
        "party_members": party_members,
        "character_options": get_character_options(campaign_id=campaign_id or None, party_id=party_id or None),
    }


def format_party_context_for_prompt(party_context: dict[str, Any] | None) -> str:
    party_context = party_context or {}
    campaign = party_context.get("campaign") or {}
    speaking = party_context.get("speaking_member") or {}
    members = party_context.get("party_members") or []

    lines = []

    campaign_id = party_context.get("campaign_id") or campaign.get("campaign_id") or ""
    if campaign_id:
        campaign_name = campaign.get("name") or campaign_id
        campaign_desc = (campaign.get("description") or "").strip()
        lines.append(f"Active campaign: {campaign_name} ({campaign_id})")
        if campaign_desc:
            lines.append(f"Campaign summary: {campaign_desc}")

    party_label = party_context.get("party_label") or party_context.get("party_id") or "Unknown party"
    lines.append(f"Party: {party_label} ({party_context.get('party_id') or 'unknown'})")

    if members:
        lines.append("Party members present:")
        for member in members:
            suffix = " — speaking now" if member.get("is_speaking") else " — present"
            lines.append(
                f"- {member.get('name')} ({member.get('race')} {member.get('class')}){suffix}"
            )
    else:
        lines.append("Party members present:")
        lines.append("- The Party (Unknown Adventurer) — speaking now")

    if speaking:
        lines.append(
            "Speaking player: "
            f"{speaking.get('name')} ({speaking.get('race')} {speaking.get('class')}), "
            f"player_id={speaking.get('player_id')}"
        )

    lines.append(
        "Use visible party composition when responding. Racial prejudice, faction bias, class assumptions, "
        "or familiarity may affect tone even when a non-speaking party member's race or class is relevant."
    )

    return "\n".join(lines)


def format_player_message_content(player_text: str, party_context: dict[str, Any] | None) -> str:
    player_text = (player_text or "").strip()
    speaking = (party_context or {}).get("speaking_member") or {}

    if not player_text:
        return player_text

    if speaking.get("is_default") or speaking.get("player_id") == DEFAULT_PLAYER_ID:
        return player_text

    name = speaking.get("name") or speaking.get("player_id") or "Player"
    race = speaking.get("race") or "Unknown"
    char_class = speaking.get("class") or "Adventurer"
    return f"[{name} ({race} {char_class})]: {player_text}"


def resolve_request_session(
    *,
    body: dict | None = None,
    args=None,
    form=None,
) -> dict[str, Any]:
    from session_context import get_request_campaign, get_request_owner, set_request_context

    body = body or {}
    args = args or {}
    form = form or {}

    owner = (
        str(body.get("owner_user_id") or form.get("owner_user_id") or args.get("owner_user_id") or "").strip()
        or get_request_owner()
        or "local"
    )
    campaign_id = (
        body.get("campaign_id")
        or form.get("campaign_id")
        or args.get("campaign_id")
        or get_request_campaign()
        or ""
    )
    campaign_id = str(campaign_id or "").strip()

    if campaign_id:
        set_request_context(owner, campaign_id)
        get_or_create_play_session(owner, campaign_id)

    ids = _current_play_ids()
    player_id = (
        body.get("player_id")
        or form.get("player_id")
        or args.get("player_id")
        or ids["player_id"]
        or DEFAULT_PLAYER_ID
    )
    party_id = (
        body.get("party_id")
        or form.get("party_id")
        or args.get("party_id")
        or ids["party_id"]
        or DEFAULT_PARTY_ID
    )
    if not campaign_id:
        campaign_id = ids["campaign_id"] or ""

    party_context = build_party_context(player_id=player_id, party_id=party_id)
    if campaign_id and not party_context.get("campaign_id"):
        party_context["campaign_id"] = campaign_id
        party_context["campaign"] = load_campaign(campaign_id)

    return {
        "player_id": party_context["player_id"],
        "party_id": party_context["party_id"],
        "campaign_id": party_context.get("campaign_id") or campaign_id or "",
        "owner_user_id": owner,
        "party_context": party_context,
    }


def get_active_session_ids() -> dict[str, str]:
    ids = _current_play_ids()
    return {
        "campaign_id": ids["campaign_id"] or "",
        "party_id": ids["party_id"] or DEFAULT_PARTY_ID,
        "player_id": ids["player_id"] or DEFAULT_PLAYER_ID,
        "owner_user_id": ids.get("owner_user_id") or "",
    }


def get_session_status() -> dict[str, Any]:
    party_context = build_party_context()
    ids = _current_play_ids()
    return {
        "owner_user_id": ids.get("owner_user_id") or "",
        "campaign_id": party_context.get("campaign_id") or "",
        "party_id": party_context.get("party_id") or DEFAULT_PARTY_ID,
        "player_id": party_context.get("player_id") or DEFAULT_PLAYER_ID,
        "active_campaign": party_context.get("campaign") or {},
        "active_party": {
            "party_id": party_context.get("party_id"),
            "label": party_context.get("party_label"),
            "members": party_context.get("party_members") or [],
        },
        "speaking_member": party_context.get("speaking_member") or {},
        "character_options": party_context.get("character_options") or [],
        "campaigns": list_campaigns(),
        "parties": list_parties(campaign_id=party_context.get("campaign_id") or None),
        "endpoints": {
            "campaigns": "/campaigns",
            "campaign_select": "/campaign/select",
            "campaign_active": "/campaign/active",
            "parties": "/parties",
            "party_characters": "/party/characters",
            "party_select": "/party/select",
            "session_status": "/session/status",
        },
    }


def init_party_campaign() -> None:
    init_party_campaign_dirs()
    _load_runtime_config()
    print(
        f"[SESSION] multi-table ready "
        f"(fallback campaign={_FALLBACK_CAMPAIGN_ID or '(none)'} "
        f"party={_FALLBACK_PARTY_ID} player={_FALLBACK_PLAYER_ID})",
        flush=True,
    )


init_party_campaign()
