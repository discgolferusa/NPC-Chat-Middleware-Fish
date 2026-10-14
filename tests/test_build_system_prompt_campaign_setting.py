"""Functional: build_system_prompt injects campaign stay-in-setting rule."""

from __future__ import annotations

import server


def test_build_system_prompt_includes_campaign_setting_rule(monkeypatch):
    monkeypatch.setattr(server, "load_global_llm_rules", lambda: "- Be concise.")
    monkeypatch.setattr(server, "format_npc_llm_rules", lambda npc: "")
    monkeypatch.setattr(server, "format_required_dm_checks_prompt_block", lambda npc, dm: "")
    monkeypatch.setattr(server, "format_location_context", lambda loc, caps=None: "A damp alley.")
    monkeypatch.setattr(
        server,
        "build_peer_location_prompt_sections",
        lambda *args, **kwargs: ("", ""),
    )
    monkeypatch.setattr(
        server,
        "get_npc_pronoun_context",
        lambda npc: {
            "gender": "female",
            "display": "she/her",
            "subject": "she",
            "object": "her",
            "possessive_adjective": "her",
            "possessive_pronoun": "hers",
            "reflexive": "herself",
        },
    )
    monkeypatch.setattr(server, "format_dm_context_for_prompt", lambda dm: "None.")
    monkeypatch.setattr(server, "build_scene_context", lambda location_id: {})
    monkeypatch.setattr(server, "format_scene_context_for_prompt", lambda ctx: "None.")
    monkeypatch.setattr(
        server,
        "get_prompt_caps",
        lambda: type("Caps", (), {})(),
    )

    npc = {
        "name": "Mira",
        "role": "Innkeeper",
        "voice": "warm",
        "personality": ["kind"],
        "public_facts": ["Runs the Blue Mug"],
        "known_facts": [],
        "secrets": [],
        "location_id": "market",
    }
    party_context = {
        "campaign_id": "greyhaven_intro",
        "party_id": "heroes",
        "party_label": "Heroes",
        "campaign": {
            "campaign_id": "greyhaven_intro",
            "name": "Greyhaven",
            "genre": "fantasy",
            "rpg_system": "D&D 5e",
            "description": "Port city intrigue.",
        },
        "party_members": [],
        "speaking_member": {},
    }

    prompt = server.build_system_prompt(
        npc,
        {"description": ""},
        memories=[],
        location={"name": "Market"},
        party_context=party_context,
    )

    assert "Campaign setting rule (REQUIRED" in prompt
    assert "Genre/setting: fantasy" in prompt
    assert "RPG system: D&D 5e" in prompt
    assert "Campaign genre/setting: fantasy" in prompt
    assert "- Be concise." in prompt


def test_build_system_prompt_omits_setting_rule_when_unset(monkeypatch):
    monkeypatch.setattr(server, "load_global_llm_rules", lambda: "- Be concise.")
    monkeypatch.setattr(server, "format_npc_llm_rules", lambda npc: "")
    monkeypatch.setattr(server, "format_required_dm_checks_prompt_block", lambda npc, dm: "")
    monkeypatch.setattr(server, "format_location_context", lambda loc, caps=None: "")
    monkeypatch.setattr(
        server,
        "build_peer_location_prompt_sections",
        lambda *args, **kwargs: ("", ""),
    )
    monkeypatch.setattr(
        server,
        "get_npc_pronoun_context",
        lambda npc: {
            "gender": "male",
            "display": "he/him",
            "subject": "he",
            "object": "him",
            "possessive_adjective": "his",
            "possessive_pronoun": "his",
            "reflexive": "himself",
        },
    )
    monkeypatch.setattr(server, "format_dm_context_for_prompt", lambda dm: "None.")
    monkeypatch.setattr(server, "build_scene_context", lambda location_id: {})
    monkeypatch.setattr(server, "format_scene_context_for_prompt", lambda ctx: "None.")
    monkeypatch.setattr(server, "get_prompt_caps", lambda: type("Caps", (), {})())

    npc = {
        "name": "Gareth",
        "role": "Guard",
        "personality": [],
        "public_facts": [],
        "known_facts": [],
        "secrets": [],
        "location_id": None,
    }
    prompt = server.build_system_prompt(
        npc,
        {},
        memories=[],
        location={},
        party_context={"campaign": {"name": "Legacy"}, "party_members": []},
    )
    assert "Campaign setting rule (REQUIRED" not in prompt


def test_player_turn_does_not_replay_share_one_secret(monkeypatch):
    """A later question must not be told to share the next secret."""
    monkeypatch.setattr(server, "load_global_llm_rules", lambda: "- Be concise.")
    monkeypatch.setattr(
        server,
        "format_npc_llm_rules",
        lambda npc: (
            "[DM Ruling: Success:id=flatter] Share exactly one secret from this NPC's "
            "secrets list that has not already been told in this conversation. "
            "If every secret was already shared, say there is nothing left to confide "
            "and do not invent one."
        ),
    )
    monkeypatch.setattr(server, "format_required_dm_checks_prompt_block", lambda npc, dm: "")
    monkeypatch.setattr(server, "format_location_context", lambda loc, caps=None: "The docks.")
    monkeypatch.setattr(
        server,
        "build_peer_location_prompt_sections",
        lambda *args, **kwargs: ("", ""),
    )
    monkeypatch.setattr(
        server,
        "get_npc_pronoun_context",
        lambda npc: {
            "gender": "female",
            "display": "she/her",
            "subject": "she",
            "object": "her",
            "possessive_adjective": "her",
            "possessive_pronoun": "hers",
            "reflexive": "herself",
        },
    )
    consequence = (
        "Share exactly one secret from this NPC's secrets list that has not already "
        "been told in this conversation. If every secret was already shared, say there "
        "is nothing left to confide and do not invent one."
    )
    monkeypatch.setattr(
        server,
        "format_dm_context_for_prompt",
        lambda dm: f"Active DM rulings:\n- [CONFIRMED] Consequence: {consequence}",
    )
    monkeypatch.setattr(server, "build_scene_context", lambda location_id: {})
    monkeypatch.setattr(server, "format_scene_context_for_prompt", lambda ctx: "None.")
    monkeypatch.setattr(server, "get_prompt_caps", lambda: type("Caps", (), {})())
    monkeypatch.setattr(
        server,
        "build_dm_injection_messages",
        lambda dm: [{
            "role": "user",
            "content": f"[DM RULING - CONFIRMED] Consequence: {consequence}",
        }],
    )

    npc = {
        "name": "Cassana",
        "role": "Thief",
        "voice": "terse",
        "personality": ["secretive"],
        "public_facts": [],
        "known_facts": [],
        "secrets": ["Knows that medical supplies were stolen from the Saranas Warehouse."],
        "location_id": "kharbranth",
    }
    encounter = f"[GM RESOLUTION] Outcome: confirmed. Consequence: {consequence}"
    messages = server.build_npc_llm_messages(
        npc,
        {},
        memories=[],
        location={},
        dm_context={"rulings": []},
        encounter_text=encounter,
        player_text="Do you have any idea what was stolen?",
    )
    prompt = "\n".join(message["content"] for message in messages)
    assert "medical supplies were stolen" not in prompt.lower()
    assert "share exactly one secret" not in prompt.lower()
    assert "does not earn one" in prompt.lower()

    followup = server.build_dm_followup_messages(
        npc,
        {},
        [],
        {},
        [],
        {"rulings": []},
        {
            "verdict": "confirmed",
            "claim_summary": "Is Cassana flattered?",
            "ruling_text": "Yes",
            "consequence_text": consequence,
            "player_id": "rasla",
        },
        encounter_text=encounter,
    )
    followup_text = "\n".join(message["content"] for message in followup)
    assert "medical supplies were stolen" in followup_text.lower()
    assert "share exactly one secret" in followup_text.lower()
