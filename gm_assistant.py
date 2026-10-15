"""GM prompt diagnosis.

Rebuilds the NPC messages a live turn would send right now, without saving a
turn or opening a DM check, then asks the content-creation model family to
explain the reply or suggest editor changes.
"""

from __future__ import annotations

from typing import Any

from flask import jsonify, request

DIAGNOSE_TEMPERATURE = 0.15
DIAGNOSE_MAX_TOKENS = 1400
MAX_YAML_CHARS = 20000
MAX_PROMPT_CHARS = 100000

_EXPLAIN_SYSTEM = """You diagnose why a tabletop NPC reply came out the way it did.

You are shown the exact messages the game would send to the NPC model right now, the player's question, and either the NPC reply or the GM's description of what went wrong.

Separate the two cases:
- The prompt allowed it. Quote the section (public facts, NPC-known facts, hidden facts, location, peer NPCs, encounter, long-term memories, DM channel, campaign setting rule, global rules, or NPC-specific rules) and the line that licensed the reply.
- The prompt did not contain it. Say the model invented or inferred it. Editing YAML will not remove a line that was never in the prompt. If nearby facts invited the inference, name them.

Memories, encounter lines, and DM rulings are session state, not NPC YAML. Say so when they are the cause, instead of telling the GM to rewrite the character.

Secrets listed only as withheld are already hidden. Do not suggest deleting secrets to hide them.
Do not propose a new live NPC reply and do not call tools. You only explain.
Do not write files or emit a patch.
""".strip()

_SUGGEST_SYSTEM = """You help a GM change NPC, location, or campaign YAML so the next reply matches what they expected.

You are shown the prompt the game would send right now, the question, what went wrong, and the YAML files the GM edits. Give narrow guidance only:
- file name
- editor field (personality, public_facts, known_facts, secrets, llm_rules, location knowledge_rules, rumors, events, or campaign rules)
- the current text that is the problem
- replacement wording the GM can type in

One field at a time. NPC-specific llm_rules override conflicting global rules; prefer those when a rule is the fix.
If the cause is a memory, encounter line, or DM ruling, say that and do not tell the GM to rewrite the character.
Do not suggest deleting secrets to hide them; withheld secrets are already hidden unless a DM check just allowed one.
Do not write files, emit a unified diff, or call tools. The GM applies the change.
""".strip()


def build_prompt_preview(
    *,
    npc_id: str,
    player_id: str,
    campaign_id: str,
    player_text: str,
    party_context: dict[str, Any] | None,
    about_npc_id: str | None = None,
    scene_description: str | None = None,
) -> dict[str, Any]:
    """Load current campaign context and build live-turn messages without writes."""
    import server
    from dm_channel import load_dm_messages, load_encounter_rulings, load_pending_checks
    from encounter_context import (
        format_encounter_prompt,
        load_encounter_memories,
        load_encounter_turns,
    )

    npc = server.load_npc(npc_id)
    location = server.load_location(npc.get("location_id"))
    scene_context = server.build_scene_context(npc.get("location_id"))
    party = party_context or server.build_party_context(player_id=player_id)
    if campaign_id and not party.get("campaign_id"):
        party = {**party, "campaign_id": campaign_id}

    with server.get_db() as conn:
        memories = load_encounter_memories(
            npc_id,
            player_id,
            campaign_id,
            limit=server.MEMORY_PROMPT_LIMIT,
            connection=conn,
        )
        encounter_turns = load_encounter_turns(
            npc_id,
            campaign_id,
            connection=conn,
        )
        dm_messages = load_dm_messages(
            npc_id,
            player_id,
            limit=10,
            campaign_id=campaign_id,
            connection=conn,
        )
        pending_checks = load_pending_checks(
            npc_id,
            player_id,
            status="pending",
            limit=None,
            campaign_id=campaign_id,
            connection=conn,
            for_update=False,
        )
        rulings = load_encounter_rulings(
            npc_id,
            player_id,
            limit=12,
            campaign_id=campaign_id,
            connection=conn,
        )

    encounter_text = format_encounter_prompt(
        encounter_turns,
        current_player_id=player_id,
        party_context=party,
    )
    dm_context = {
        "campaign_id": campaign_id,
        "messages": dm_messages[-10:],
        "pending_checks": pending_checks[-10:],
        "rulings": rulings,
    }
    messages = messages_from_preview_parts(
        npc=npc,
        location=location,
        memories=memories,
        dm_context=dm_context,
        party_context=party,
        scene_context=scene_context,
        player_text=player_text,
        about_npc_id=about_npc_id,
        scene_description=scene_description,
        encounter_text=encounter_text,
    )
    system_prompt = next(
        (str(item.get("content") or "") for item in messages if item.get("role") == "system"),
        "",
    )
    return {
        "npc_id": npc.get("npc_id") or npc.get("id") or npc_id,
        "npc_name": npc.get("name") or npc_id,
        "player_id": player_id,
        "campaign_id": campaign_id,
        "location_id": npc.get("location_id") or "",
        "messages": messages,
        "sections": section_index(
            npc=npc,
            location=location if isinstance(location, dict) else {},
            memories=memories,
            dm_context=dm_context,
            encounter_text=encounter_text,
            system_prompt=system_prompt,
        ),
    }


def messages_from_preview_parts(
    *,
    npc: dict,
    location: dict | None,
    memories: list[str] | None,
    dm_context: dict | None,
    party_context: dict | None,
    scene_context: dict | None,
    player_text: str,
    about_npc_id: str | None,
    scene_description: str | None,
    encounter_text: str | None,
) -> list[dict[str, str]]:
    """Same assembly as a live turn: encounter text is always set, so chat history is not appended."""
    import server

    return server.build_npc_llm_messages(
        npc,
        {"description": (scene_description or "").strip()},
        memories or [],
        location,
        player_text=player_text,
        dm_context=dm_context,
        party_context=party_context,
        scene_context=scene_context,
        about_npc_id=(about_npc_id or "").strip() or None,
        encounter_text=encounter_text if encounter_text is not None else "",
        allow_secret_disclosure=False,
    )


def section_index(
    *,
    npc: dict,
    location: dict,
    memories: list[str],
    dm_context: dict,
    encounter_text: str,
    system_prompt: str,
) -> dict[str, Any]:
    secrets = [str(item).strip() for item in npc.get("secrets") or [] if str(item or "").strip()]
    prompt = system_prompt or ""
    return {
        "public_facts": [str(item) for item in npc.get("public_facts") or []],
        "known_facts": [str(item) for item in npc.get("known_facts") or []],
        "secrets_withheld": bool(secrets) and "withheld" in prompt.lower(),
        "secret_count": len(secrets),
        "location_id": str(location.get("id") or npc.get("location_id") or ""),
        "location_name": str(location.get("name") or ""),
        "has_peer_directory": "People commonly known to be around here:" in prompt,
        "has_peer_facts": "Peer knowledge rules" in prompt,
        "memory_count": len(memories or []),
        "encounter_present": bool((encounter_text or "").strip())
        and "No encounter lines are on record" not in (encounter_text or ""),
        "dm_message_count": len((dm_context or {}).get("messages") or []),
        "pending_check_count": len((dm_context or {}).get("pending_checks") or []),
        "ruling_count": len((dm_context or {}).get("rulings") or []),
        "has_npc_rules": "NPC-specific rules" in prompt,
        "has_global_rules": "Rules:" in prompt,
    }


def build_gm_assistant_messages(
    *,
    mode: str,
    preview_messages: list[dict[str, Any]],
    question: str,
    npc_reply: str | None,
    mismatch_note: str | None,
    yaml_documents: list[dict[str, Any]] | None,
    followup: str | None,
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Prompt for the diagnosis model. Roles are only system, user, and assistant."""
    normalized = (mode or "explain").strip().lower()
    if normalized not in {"explain", "suggest"}:
        raise ValueError("mode must be explain or suggest.")

    system = _SUGGEST_SYSTEM if normalized == "suggest" else _EXPLAIN_SYSTEM
    case_lines = [
        "NPC prompt messages the game would send right now:",
        _format_preview_messages(preview_messages),
        "",
        f"Player question:\n{(question or '').strip() or '(not provided)'}",
    ]
    reply = (npc_reply or "").strip()
    mismatch = (mismatch_note or "").strip()
    if reply:
        case_lines.append(f"\nNPC reply the GM is asking about:\n{reply}")
    if mismatch:
        case_lines.append(f"\nWhat did not fit:\n{mismatch}")
    documents = yaml_documents or []
    if documents:
        case_lines.append("\nYAML the GM can edit:")
        for document in documents:
            kind = str(document.get("kind") or "file")
            filename = str(document.get("filename") or document.get("id") or "untitled")
            text = str(document.get("text") or "")
            if len(text) > MAX_YAML_CHARS:
                text = text[:MAX_YAML_CHARS].rstrip() + "\n...[truncated]"
            case_lines.append(f"\n--- {kind} {filename} ---\n{text}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(case_lines).strip()},
    ]
    for item in history or []:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})

    extra = (followup or "").strip()
    if extra:
        messages.append({"role": "user", "content": extra})
    elif normalized == "suggest":
        messages.append({
            "role": "user",
            "content": "What should I change in the editor so the next reply matches what I expected?",
        })
    return messages


def complete_diagnosis(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
    from llm_router import PURPOSE_GM_ASSISTANT, generate_llm_response_with_usage

    text, usage = generate_llm_response_with_usage(
        messages,
        purpose=PURPOSE_GM_ASSISTANT,
        temperature=DIAGNOSE_TEMPERATURE,
        max_tokens=DIAGNOSE_MAX_TOKENS,
    )
    return text, usage if isinstance(usage, dict) else {}


def handle_prompt_preview():
    import server
    from party_campaign import resolve_request_session

    body = request.get_json(force=True, silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required."}), 400

    npc_id = str(body.get("npc_id") or "").strip()
    text = str(body.get("text") or "").strip()
    if not npc_id or not text:
        return jsonify({"error": "npc_id and text are required."}), 400

    player_id = str(body.get("player_id") or "").strip() or "dm_console"
    session_body = {**body, "player_id": player_id}
    session = resolve_request_session(body=session_body, args=request.args)
    resolved_player = str(session.get("player_id") or player_id).strip() or player_id
    campaign_id = str(session.get("campaign_id") or body.get("campaign_id") or "").strip()

    try:
        preview = build_prompt_preview(
            npc_id=npc_id,
            player_id=resolved_player,
            campaign_id=campaign_id,
            player_text=text,
            party_context=session.get("party_context"),
            about_npc_id=str(body.get("about_npc_id") or "").strip() or None,
            scene_description=str(body.get("scene") or ""),
        )
    except FileNotFoundError:
        return jsonify({"error": f"Unknown NPC: {npc_id}"}), 404
    except Exception as ex:
        print(f"[GM-ASSISTANT] preview failed npc={npc_id}: {ex!r}", flush=True)
        return jsonify({"error": "Could not build the NPC prompt."}), 500

    preview["player_id"] = resolved_player
    return jsonify(preview)


def handle_diagnose():
    body = request.get_json(force=True, silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required."}), 400

    preview_messages = body.get("messages")
    if not isinstance(preview_messages, list) or not preview_messages:
        return jsonify({"error": "messages from the prompt preview are required."}), 400

    question = str(body.get("question") or "").strip()
    npc_reply = str(body.get("npc_reply") or "").strip()
    mismatch = str(body.get("mismatch_note") or "").strip()
    if not question:
        return jsonify({"error": "question is required."}), 400
    if not npc_reply and not mismatch:
        return jsonify({"error": "npc_reply or mismatch_note is required."}), 400

    mode = str(body.get("mode") or "explain").strip().lower()
    documents = body.get("yaml_documents")
    if documents is None:
        documents = []
    if not isinstance(documents, list):
        return jsonify({"error": "yaml_documents must be a list."}), 400
    history = body.get("history")
    if history is None:
        history = []
    if not isinstance(history, list):
        return jsonify({"error": "history must be a list."}), 400

    try:
        messages = build_gm_assistant_messages(
            mode=mode,
            preview_messages=preview_messages,
            question=question,
            npc_reply=npc_reply,
            mismatch_note=mismatch,
            yaml_documents=documents,
            followup=str(body.get("followup") or ""),
            history=history,
        )
        text, usage = complete_diagnosis(messages)
    except ValueError as ex:
        return jsonify({"error": str(ex)}), 400
    except Exception as ex:
        print(f"[GM-ASSISTANT] diagnose failed: {ex!r}", flush=True)
        return jsonify({"error": "The diagnosis model did not answer."}), 502

    return jsonify({
        "mode": mode,
        "text": text,
        "usage": usage,
    })


def _format_preview_messages(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    used = 0
    for item in messages:
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "")
        block = f"[{role}]\n{content}"
        if used + len(block) > MAX_PROMPT_CHARS:
            remaining = MAX_PROMPT_CHARS - used
            if remaining > 0:
                chunks.append(block[:remaining].rstrip() + "\n...[truncated]")
            break
        chunks.append(block)
        used += len(block)
    return "\n\n".join(chunks)
