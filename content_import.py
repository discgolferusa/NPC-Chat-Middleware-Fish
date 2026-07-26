"""
Campaign document import: turn a pasted campaign document (module notes,
session prep, wiki export, etc.) into structured campaign/location/NPC data
using multi-pass LLM calls against the "content_creation" purpose model.

This module never writes files or touches content packs — it only returns
structured dicts (shaped like the campaign/location/npc YAML schemas used
elsewhere in this repo) for the caller to review and persist.

Pipeline:
  1. Outline pass — one LLM call that reads the whole document and proposes
     the campaign, a location list, and an NPC list (name/role/locations).
  2. Location detail pass — batched LLM calls (<= MAX_LOCATIONS_PER_BATCH per
     call) that fill in summary/current_events/important_locations/rumors/
     local_tone/knowledge_rules for each location.
  3. NPC detail pass — batched LLM calls (<= MAX_NPCS_PER_BATCH per call) that
     fill in personality/public_facts/known_facts/secrets/llm_rules/pronouns/
     race/gender for each NPC.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_router import generate_llm_response

MAX_INPUT_CHARS = 120_000
MAX_LOCATIONS_PER_BATCH = 8
MAX_NPCS_PER_BATCH = 5
IMPORT_TEMPERATURE = 0.2
IMPORT_MAX_TOKENS = 4096

VALID_MODES = frozenset({"create_new", "merge"})

_FENCE_PATTERN = re.compile(r"```(?:json|yaml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SNAKE_INVALID = re.compile(r"[^a-z0-9]+")
_SNAKE_COLLAPSE = re.compile(r"_+")


class ContentImportError(Exception):
    """Raised when the document cannot be analyzed into structured content."""


# --------------------------------------------------------------------------
# JSON / text helpers
# --------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _find_balanced_json(text: str, open_ch: str, close_ch: str) -> Any | None:
    start = text.find(open_ch)
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
        start = text.find(open_ch, start + 1)
    return None


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM completion.

    Handles markdown code fences, leading/trailing prose around the JSON, and
    falls back to scanning for the first balanced {...} or [...] block.
    """
    if not text or not text.strip():
        raise ContentImportError("Empty LLM response")

    candidates = [_strip_code_fences(text), text.strip()]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    for candidate in candidates:
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            result = _find_balanced_json(candidate, open_ch, close_ch)
            if result is not None:
                return result

    raise ContentImportError(f"Could not extract JSON from LLM response: {text[:200]!r}")


def snake_case(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _SNAKE_INVALID.sub("_", text)
    text = _SNAKE_COLLAPSE.sub("_", text).strip("_")
    return text or "unnamed"


def _unique_id(base: str, taken: set[str]) -> str:
    base = base or "unnamed"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)] if items else []


def _coerce_container(data: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        # Some models collapse a single-item batch into the bare object.
        if "id" in data:
            return [data]
    return []


def _call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    warnings: list[str],
    step: str,
) -> Any:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = generate_llm_response(
        messages,
        purpose="content_creation",
        max_tokens=IMPORT_MAX_TOKENS,
        temperature=IMPORT_TEMPERATURE,
    )
    try:
        return extract_json(raw)
    except ContentImportError as ex:
        warnings.append(f"{step}: {ex}")
        raise


_SYSTEM_PROMPT = (
    "You are a tabletop RPG campaign content structuring assistant. You read campaign "
    "documents (adventure modules, session prep notes, wiki exports) and convert them into "
    "strict JSON matching the schema the user requests. Respond with ONLY the JSON value — "
    "no markdown code fences, no commentary before or after it."
)


# --------------------------------------------------------------------------
# Pass 1: outline
# --------------------------------------------------------------------------


def _build_outline_prompt(
    text: str,
    mode: str,
    existing_campaign_id: str | None,
    existing_location_ids: list[str],
    existing_npc_names: list[str],
) -> str:
    if mode == "merge" and existing_campaign_id:
        mode_guidance = (
            f"This document is being merged into the EXISTING campaign \"{existing_campaign_id}\". "
            f"Set campaign.campaign_id to exactly \"{existing_campaign_id}\".\n"
        )
        if existing_location_ids:
            mode_guidance += (
                "These location ids already exist in the campaign: "
                f"{', '.join(existing_location_ids)}. If the document clearly describes one of "
                "these same locations, reuse that exact id; otherwise invent a new distinct id.\n"
            )
        if existing_npc_names:
            mode_guidance += (
                "These NPC names already exist in the campaign: "
                f"{', '.join(existing_npc_names)}. If the document clearly describes one of these "
                "same NPCs, reuse that exact name; otherwise invent a new distinct name.\n"
            )
    else:
        mode_guidance = (
            "This is a brand new campaign. Invent a short, memorable campaign_id and name from "
            "the document's title or setting if one is not explicitly given.\n"
        )

    return (
        "Read the campaign document text below and produce an outline as JSON with exactly "
        "this shape:\n\n"
        "{\n"
        '  "campaign": {"campaign_id": "snake_case_id", "name": "Display Name", '
        '"description": "1-3 sentence summary"},\n'
        '  "locations": [{"id": "snake_case_id", "name": "Display Name", '
        '"type": "city|town|dungeon|forest|building|region|other"}],\n'
        '  "npcs": [{"name": "Display Name", "role": "short role or title", '
        '"first_location_id": "snake_case id of the location where this NPC is first '
        'encountered", "other_location_ids": ["snake_case_id", ...]}]\n'
        "}\n\n"
        "Rules:\n"
        "- Use snake_case for every id (lowercase letters, digits, underscores only).\n"
        "- Every id used as an NPC's first_location_id or in other_location_ids MUST also "
        "appear in the locations list.\n"
        "- Include every named location and every named NPC you can find, even minor ones.\n"
        "- other_location_ids should only be non-empty for NPCs who travel or appear in more "
        "than one place in the text.\n"
        f"{mode_guidance}"
        "- Output ONLY the JSON object described above. No prose, no markdown fences.\n\n"
        "Campaign document text:\n"
        '"""\n'
        f"{text}\n"
        '"""'
    )


def _normalize_outline(
    outline: dict[str, Any],
    *,
    mode: str,
    existing_campaign_id: str | None,
    existing_location_ids: list[str],
    existing_npc_names: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    campaign_raw = outline.get("campaign") if isinstance(outline.get("campaign"), dict) else {}

    if mode == "merge" and existing_campaign_id:
        campaign_id = existing_campaign_id.strip()
    else:
        campaign_id = snake_case(campaign_raw.get("campaign_id") or campaign_raw.get("name") or "campaign")

    campaign = {
        "campaign_id": campaign_id,
        "name": str(campaign_raw.get("name") or campaign_id.replace("_", " ").title()).strip(),
        "description": str(campaign_raw.get("description") or "").strip(),
    }

    taken_location_ids: set[str] = {loc_id.strip() for loc_id in existing_location_ids if loc_id and loc_id.strip()}
    id_remap: dict[str, str] = {}
    locations: list[dict[str, Any]] = []

    for raw_loc in _as_list(outline.get("locations")):
        if not isinstance(raw_loc, dict):
            continue
        name = str(raw_loc.get("name") or "").strip()
        raw_id = str(raw_loc.get("id") or name or "location").strip()
        base_id = snake_case(raw_id)
        final_id = base_id
        if final_id in taken_location_ids:
            final_id = _unique_id(base_id, taken_location_ids)
            warnings.append(
                f"Location id '{base_id}' collided with an existing location; renamed to '{final_id}'."
            )
        else:
            taken_location_ids.add(final_id)
        id_remap[raw_id] = final_id
        id_remap[base_id] = final_id
        locations.append(
            {
                "id": final_id,
                "name": name or final_id.replace("_", " ").title(),
                "type": str(raw_loc.get("type") or "other").strip() or "other",
            }
        )

    known_location_ids = {loc["id"] for loc in locations}

    def _resolve_location_ref(ref: str) -> str | None:
        ref = str(ref or "").strip()
        if not ref:
            return None
        if ref in id_remap:
            return id_remap[ref]
        candidate = snake_case(ref)
        if candidate in id_remap:
            return id_remap[candidate]
        if candidate in known_location_ids:
            return candidate
        return None

    taken_npc_ids: set[str] = {snake_case(n) for n in existing_npc_names if n and n.strip()}
    npcs: list[dict[str, Any]] = []

    for raw_npc in _as_list(outline.get("npcs")):
        if not isinstance(raw_npc, dict):
            continue
        name = str(raw_npc.get("name") or "").strip()
        if not name:
            continue
        base_id = snake_case(name)
        final_id = base_id
        if final_id in taken_npc_ids:
            final_id = _unique_id(base_id, taken_npc_ids)
            warnings.append(
                f"NPC id '{base_id}' collided with an existing NPC name; renamed to '{final_id}'."
            )
        else:
            taken_npc_ids.add(final_id)

        first_location_id = _resolve_location_ref(raw_npc.get("first_location_id"))
        other_location_ids: list[str] = []
        for ref in _as_list(raw_npc.get("other_location_ids")):
            resolved = _resolve_location_ref(ref)
            if resolved and resolved != first_location_id and resolved not in other_location_ids:
                other_location_ids.append(resolved)

        if raw_npc.get("first_location_id") and not first_location_id:
            warnings.append(
                f"NPC '{name}': could not resolve first_location_id "
                f"'{raw_npc.get('first_location_id')}' to a known location."
            )

        npcs.append(
            {
                "id": final_id,
                "name": name,
                "role": str(raw_npc.get("role") or "").strip(),
                "first_location_id": first_location_id or (locations[0]["id"] if locations else ""),
                "other_location_ids": other_location_ids,
            }
        )

    return campaign, locations, npcs


# --------------------------------------------------------------------------
# Pass 2: location detail
# --------------------------------------------------------------------------


def _build_location_detail_prompt(batch: list[dict[str, Any]], text: str, campaign: dict[str, Any]) -> str:
    location_list = "\n".join(f"- id: {loc['id']} | name: {loc['name']} | type: {loc['type']}" for loc in batch)
    return (
        f'Campaign: "{campaign["name"]}" ({campaign["campaign_id"]}).\n\n'
        "Fill in details for EACH of the following locations using the campaign document text "
        'below. Respond with exactly {"locations": [ ... ]} where each array item has this '
        "shape:\n\n"
        "{\n"
        '  "id": "<exact id from the list below>",\n'
        '  "summary": "2-4 sentence overview of the location",\n'
        '  "current_events": ["ongoing situations relevant right now, 2-5 items"],\n'
        '  "important_locations": [{"name": "sub-location name", "description": "1 sentence", '
        '"public_knowledge": "what ordinary people believe or know about it"}],\n'
        '  "rumors": ["uncertain rumors circulating, 2-5 items"],\n'
        '  "local_tone": ["3-6 single-word or short-phrase mood/atmosphere descriptors"],\n'
        '  "knowledge_rules": ["1-3 short rules about what NPCs here generally do or do not '
        'know, e.g. \'NPCs here do not automatically know secrets belonging to other NPCs.\'"]\n'
        "}\n\n"
        "If the document does not give enough detail for a field, invent a short, reasonable, "
        "in-setting default rather than leaving it empty.\n\n"
        "Locations to fill in:\n"
        f"{location_list}\n\n"
        "Campaign document text:\n"
        '"""\n'
        f"{text}\n"
        '"""\n\n'
        'Output ONLY {"locations": [...]}. No prose, no markdown fences.'
    )


def _merge_location_detail(stub: dict[str, Any], detail: dict[str, Any], campaign_id: str) -> dict[str, Any]:
    important_locations = []
    for item in _as_list(detail.get("important_locations")):
        if isinstance(item, dict):
            important_locations.append(
                {
                    "name": str(item.get("name") or "").strip(),
                    "description": str(item.get("description") or "").strip(),
                    "public_knowledge": str(item.get("public_knowledge") or "").strip(),
                }
            )
        elif isinstance(item, str) and item.strip():
            important_locations.append({"name": item.strip(), "description": "", "public_knowledge": ""})

    return {
        "id": stub["id"],
        "campaign_id": campaign_id,
        "name": stub["name"],
        "type": stub["type"],
        "summary": str(detail.get("summary") or "").strip(),
        "current_events": _as_str_list(detail.get("current_events")),
        "important_locations": important_locations,
        "rumors": _as_str_list(detail.get("rumors")),
        "local_tone": _as_str_list(detail.get("local_tone")),
        "knowledge_rules": _as_str_list(detail.get("knowledge_rules")),
    }


# --------------------------------------------------------------------------
# Pass 3: NPC detail
# --------------------------------------------------------------------------


def _build_npc_detail_prompt(batch: list[dict[str, Any]], text: str, campaign: dict[str, Any]) -> str:
    npc_lines = []
    for npc in batch:
        locations_note = npc["first_location_id"]
        if npc["other_location_ids"]:
            locations_note += f" (also appears at: {', '.join(npc['other_location_ids'])})"
        npc_lines.append(
            f"- id: {npc['id']} | name: {npc['name']} | role: {npc['role'] or 'unspecified'} | "
            f"location(s): {locations_note}"
        )
    npc_list = "\n".join(npc_lines)

    return (
        f'Campaign: "{campaign["name"]}" ({campaign["campaign_id"]}).\n\n'
        "Fill in roleplay details for EACH of the following NPCs using the campaign document "
        'text below. Respond with exactly {"npcs": [ ... ]} where each array item has this '
        "shape:\n\n"
        "{\n"
        '  "id": "<exact id from the list below>",\n'
        '  "pronouns": "he/him | she/her | they/them | <other, e.g. \'ze/zir\'>",\n'
        '  "race": "npc\'s race or species if known, otherwise a reasonable guess, otherwise '
        '\\"unknown\\"",\n'
        '  "gender": "male|female|nonbinary|unspecified",\n'
        '  "personality": ["3-6 short personality traits"],\n'
        '  "public_facts": ["facts about this NPC that OTHER NPCs and common folk already know '
        'and might mention, 2-5 items"],\n'
        '  "known_facts": ["facts this NPC personally knows and MAY share if asked or relevant, '
        '2-6 items"],\n'
        '  "secrets": ["facts this NPC will NOT willingly divulge, 0-5 items"],\n'
        '  "llm_rules": ["short behavioral rules for roleplaying this NPC, 2-6 items"]\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "- public_facts = things OTHER characters already know about this NPC (reputation, "
        "visible role, public history) — NOT things only the NPC themself knows.\n"
        "- known_facts = things this NPC personally knows and may share in conversation.\n"
        "- secrets = things this NPC will NOT willingly reveal to the party.\n"
        "- If an NPC has ANY secrets, you MUST include at least one llm_rule of this exact "
        'form: "If asked directly about <topic>, do not reveal it. Respond in character, then '
        'end your reply with [DM CHECK: <short description of what the player is trying to '
        'learn>] so the DM can decide if it comes out." Tailor <topic> and the bracketed '
        "description to this NPC's actual secrets.\n"
        "- If an NPC appears at more than one location (see location(s) below), any fact, "
        "secret, or rule that should only apply at a specific location must be prefixed "
        'exactly like this: "Only when scene location is <location_id>: <fact or rule>".\n'
        "- Do not invent facts that contradict the source text. If the text gives no detail "
        "for a field, provide a short reasonable in-setting default, except secrets (use an "
        "empty list if the NPC clearly has nothing to hide).\n\n"
        "NPCs to fill in:\n"
        f"{npc_list}\n\n"
        "Campaign document text:\n"
        '"""\n'
        f"{text}\n"
        '"""\n\n'
        'Output ONLY {"npcs": [...]}. No prose, no markdown fences.'
    )


def _merge_npc_detail(stub: dict[str, Any], detail: dict[str, Any], campaign_id: str, warnings: list[str]) -> dict[str, Any]:
    secrets = _as_str_list(detail.get("secrets"))
    llm_rules = _as_str_list(detail.get("llm_rules"))

    if secrets and not any("[DM CHECK:" in rule for rule in llm_rules):
        llm_rules.append(
            "If asked directly about information you are keeping secret, do not reveal it. "
            "Respond in character, then end your reply with [DM CHECK: the player is probing "
            "one of this NPC's secrets] so the DM can decide if it comes out."
        )
        warnings.append(f"NPC '{stub['name']}': added a default [DM CHECK] rule for its secrets.")

    return {
        "id": stub["id"],
        "campaign_id": campaign_id,
        "name": stub["name"],
        "role": stub["role"],
        "location_id": stub["first_location_id"],
        "other_location_ids": stub["other_location_ids"],
        "gender": str(detail.get("gender") or "unspecified").strip() or "unspecified",
        "race": str(detail.get("race") or "unknown").strip() or "unknown",
        "pronouns": str(detail.get("pronouns") or "they/them").strip() or "they/them",
        "personality": _as_str_list(detail.get("personality")),
        "public_facts": _as_str_list(detail.get("public_facts")),
        "known_facts": _as_str_list(detail.get("known_facts")),
        "secrets": secrets,
        "llm_rules": llm_rules,
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _truncate_input(text: str, warnings: list[str]) -> str:
    text = text or ""
    if len(text) > MAX_INPUT_CHARS:
        warnings.append(
            f"Input document was {len(text)} characters; truncated to the first "
            f"{MAX_INPUT_CHARS} characters for analysis."
        )
        return text[:MAX_INPUT_CHARS]
    return text


def analyze_campaign_document(
    text: str,
    mode: str,
    existing_campaign_id: str | None = None,
    existing_location_ids: list[str] | None = None,
    existing_npc_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze a campaign document (module notes, session prep, wiki export) and
    return structured campaign/location/NPC data via multi-pass LLM calls.

    Returns: {"campaign": {...}, "locations": [...], "npcs": [...], "warnings": [...]}
    Raises ContentImportError if the document cannot be analyzed at all.
    """
    warnings: list[str] = []

    mode = (mode or "create_new").strip().lower()
    if mode not in VALID_MODES:
        warnings.append(f"Unknown mode '{mode}'; defaulting to 'create_new'.")
        mode = "create_new"

    existing_location_ids = [str(x).strip() for x in (existing_location_ids or []) if str(x).strip()]
    existing_npc_names = [str(x).strip() for x in (existing_npc_names or []) if str(x).strip()]
    existing_campaign_id = (existing_campaign_id or "").strip() or None

    if mode == "merge" and not existing_campaign_id:
        warnings.append("mode='merge' but no existing_campaign_id was provided; treating as create_new.")
        mode = "create_new"

    if not text or not text.strip():
        raise ContentImportError("No document text provided")

    document_text = _truncate_input(text, warnings)

    outline_prompt = _build_outline_prompt(
        document_text, mode, existing_campaign_id, existing_location_ids, existing_npc_names
    )

    try:
        outline_raw = _call_llm_json(_SYSTEM_PROMPT, outline_prompt, warnings=warnings, step="outline")
    except Exception as ex:
        raise ContentImportError(f"Outline pass failed: {ex}") from ex

    if not isinstance(outline_raw, dict):
        raise ContentImportError("Outline pass did not return a JSON object")

    campaign, location_stubs, npc_stubs = _normalize_outline(
        outline_raw,
        mode=mode,
        existing_campaign_id=existing_campaign_id,
        existing_location_ids=existing_location_ids,
        existing_npc_names=existing_npc_names,
        warnings=warnings,
    )

    locations_by_id: dict[str, dict[str, Any]] = {}
    for batch in _batches(location_stubs, MAX_LOCATIONS_PER_BATCH):
        prompt = _build_location_detail_prompt(batch, document_text, campaign)
        try:
            detail_raw = _call_llm_json(_SYSTEM_PROMPT, prompt, warnings=warnings, step="location detail")
            detail_items = _coerce_container(detail_raw, "locations")
            details_by_id = {str(item.get("id") or "").strip(): item for item in detail_items}
        except Exception as ex:
            warnings.append(f"Location detail batch failed ({[loc['id'] for loc in batch]}): {ex}")
            details_by_id = {}

        for stub in batch:
            detail = details_by_id.get(stub["id"], {})
            if not detail:
                warnings.append(f"Location '{stub['id']}': no detail returned; using minimal defaults.")
            locations_by_id[stub["id"]] = _merge_location_detail(stub, detail, campaign["campaign_id"])

    npcs_by_id: dict[str, dict[str, Any]] = {}
    for batch in _batches(npc_stubs, MAX_NPCS_PER_BATCH):
        prompt = _build_npc_detail_prompt(batch, document_text, campaign)
        try:
            detail_raw = _call_llm_json(_SYSTEM_PROMPT, prompt, warnings=warnings, step="npc detail")
            detail_items = _coerce_container(detail_raw, "npcs")
            details_by_id = {str(item.get("id") or "").strip(): item for item in detail_items}
        except Exception as ex:
            warnings.append(f"NPC detail batch failed ({[npc['id'] for npc in batch]}): {ex}")
            details_by_id = {}

        for stub in batch:
            detail = details_by_id.get(stub["id"], {})
            if not detail:
                warnings.append(f"NPC '{stub['id']}': no detail returned; using minimal defaults.")
            npcs_by_id[stub["id"]] = _merge_npc_detail(stub, detail, campaign["campaign_id"], warnings)

    return {
        "campaign": campaign,
        "locations": [locations_by_id[stub["id"]] for stub in location_stubs],
        "npcs": [npcs_by_id[stub["id"]] for stub in npc_stubs],
        "warnings": warnings,
    }
