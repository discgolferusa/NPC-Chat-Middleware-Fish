"""GM prompt preview and diagnosis assistant."""

from __future__ import annotations

import gm_assistant
import llm_router


def _sample_npc() -> dict:
    return {
        "name": "Mira",
        "npc_id": "mira",
        "role": "Innkeeper",
        "voice": "warm",
        "personality": ["kind"],
        "public_facts": ["Runs the Blue Mug"],
        "known_facts": ["The cellar key is brass"],
        "secrets": ["The cellar hides a tunnel"],
        "location_id": "market",
        "llm_rules": ["Never mention the tunnel."],
    }


def test_preview_messages_match_live_builder(api, monkeypatch):
    _client, server, _headers = api
    monkeypatch.setattr(server, "load_global_llm_rules", lambda: "- Stay in character.")
    npc = _sample_npc()
    location = {"id": "market", "name": "Market"}
    memories = ["The party paid their tab."]
    dm_context = {"campaign_id": "camp", "messages": [], "pending_checks": [], "rulings": []}
    party = {
        "campaign_id": "camp",
        "player_id": "dm_console",
        "campaign": {"name": "Greyhaven"},
        "party_members": [],
        "speaking_member": {"name": "GM", "player_id": "dm_console"},
    }
    scene = {"time_of_day": "night"}
    question = "Where is the cellar key?"

    messages = gm_assistant.messages_from_preview_parts(
        npc=npc,
        location=location,
        memories=memories,
        dm_context=dm_context,
        party_context=party,
        scene_context=scene,
        player_text=question,
        about_npc_id=None,
        scene_description="",
        encounter_text="",
    )
    expected = server.build_npc_llm_messages(
        npc,
        {"description": ""},
        memories,
        location,
        player_text=question,
        dm_context=dm_context,
        party_context=party,
        scene_context=scene,
        about_npc_id=None,
        encounter_text="",
        allow_secret_disclosure=False,
    )

    assert messages == expected
    assert messages[-1]["role"] == "user"
    assert "cellar key" in messages[-1]["content"]

    with_history = server.build_npc_llm_messages(
        npc,
        {"description": ""},
        memories,
        location,
        history=[{"role": "user", "content": "OLD HISTORY"}],
        player_text=question,
        dm_context=dm_context,
        party_context=party,
        scene_context=scene,
        encounter_text=None,
    )
    assert any("OLD HISTORY" in item["content"] for item in with_history)
    assert all("OLD HISTORY" not in item["content"] for item in messages)


def test_diagnose_prompt_explain_vs_suggest_includes_yaml_and_no_tools():
    preview = [
        {"role": "system", "content": "NPC name: Mira\nPublic facts:\n- Runs the Blue Mug"},
        {"role": "user", "content": "Where is the key?"},
    ]
    yaml_documents = [{
        "kind": "npc",
        "id": "mira",
        "filename": "mira.yaml",
        "text": "name: Mira\npublic_facts:\n- Runs the Blue Mug\n",
    }]

    explain = gm_assistant.build_gm_assistant_messages(
        mode="explain",
        preview_messages=preview,
        question="Where is the key?",
        npc_reply="Under the bar, next to the tunnel.",
        mismatch_note="She revealed the tunnel.",
        yaml_documents=yaml_documents,
        followup="",
        history=[],
    )
    suggest = gm_assistant.build_gm_assistant_messages(
        mode="suggest",
        preview_messages=preview,
        question="Where is the key?",
        npc_reply="Under the bar, next to the tunnel.",
        mismatch_note="",
        yaml_documents=yaml_documents,
        followup="How do I stop her mentioning the tunnel?",
        history=[{"role": "assistant", "content": "The reply invented the tunnel."}],
    )

    assert explain[0]["content"] != suggest[0]["content"]
    assert "invented or inferred" in explain[0]["content"]
    assert "editor field" in suggest[0]["content"]
    assert "mira.yaml" in explain[1]["content"]
    assert "mira.yaml" in suggest[1]["content"]
    assert suggest[-1]["content"] == "How do I stop her mentioning the tunnel?"
    assert any(item["content"] == "The reply invented the tunnel." for item in suggest)
    for batch in (explain, suggest):
        assert all(item["role"] in {"system", "user", "assistant"} for item in batch)
        assert not any(item["role"] in {"tool", "function"} for item in batch)


def test_gm_assistant_inherits_content_creation_model(monkeypatch):
    monkeypatch.setattr(llm_router, "GM_ASSISTANT_MODEL_ID", "")
    monkeypatch.setattr(llm_router, "CONTENT_CREATION_MODEL_ID", llm_router.CURRENT_LLM_MODEL_ID)
    assert llm_router._get_purpose_model_id("gm_assistant") == llm_router._get_purpose_model_id(
        "content_creation"
    )
    assert llm_router._is_content_family_purpose("gm_assistant")
    assert llm_router.model_supports_purpose("local", "gm_assistant")


def _counts(server):
    with server.get_db() as conn:
        turns = conn.execute("SELECT COUNT(*) AS n FROM conversation_turns").fetchone()["n"]
        checks = conn.execute("SELECT COUNT(*) AS n FROM dm_pending_checks").fetchone()["n"]
    return turns, checks


def test_prompt_preview_returns_messages_without_saving(api, monkeypatch):
    client, server, headers = api
    before = _counts(server)

    def refuse(*_args, **_kwargs):
        raise AssertionError("prompt preview must not mutate play state")

    monkeypatch.setattr(server, "save_turn", refuse)
    monkeypatch.setattr(server, "process_player_message", refuse)
    monkeypatch.setattr(server, "prepare_turn_database_phase", refuse)

    response = client.post(
        "/npc-prompt-preview",
        headers=headers,
        json={
            "npc_id": "captain_varra",
            "campaign_id": "greyhaven_intro",
            "player_id": "dm_console",
            "text": "Who ordered the aqueduct closed?",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["player_id"] == "dm_console"
    assert payload["messages"][0]["role"] == "system"
    assert "aqueduct" in payload["messages"][0]["content"].lower()
    assert "Who ordered the aqueduct closed?" in payload["messages"][-1]["content"]
    assert "public_facts" in payload["sections"]
    assert _counts(server) == before


def test_diagnose_route_uses_assistant_purpose_without_saving(api, monkeypatch):
    client, server, headers = api
    before = _counts(server)
    seen = {}

    def fake_complete(messages, **kwargs):
        seen["purpose"] = kwargs.get("purpose")
        seen["messages"] = messages
        return "Public facts told her the aqueduct was closed.", {"total_tokens": 4}

    monkeypatch.setattr(llm_router, "generate_llm_response_with_usage", fake_complete)
    monkeypatch.setattr(server, "save_turn", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("save")))

    response = client.post(
        "/gm-assistant/diagnose",
        headers=headers,
        json={
            "mode": "explain",
            "question": "Who closed the aqueduct?",
            "npc_reply": "The mayor did.",
            "messages": [
                {"role": "system", "content": "Public facts:\n- The aqueduct is closed."},
                {"role": "user", "content": "Who closed the aqueduct?"},
            ],
            "yaml_documents": [{
                "kind": "npc",
                "id": "captain_varra",
                "filename": "captain_varra.yaml",
                "text": "public_facts:\n- The aqueduct is closed.\n",
            }],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "aqueduct" in payload["text"].lower()
    assert seen["purpose"] == "gm_assistant"
    assert "captain_varra.yaml" in seen["messages"][1]["content"]
    assert _counts(server) == before
