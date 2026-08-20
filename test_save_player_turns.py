"""Unit and functional tests for conversation turn persistence."""

from __future__ import annotations


def test_save_player_npc_conversation_turns_can_skip_user(api):
    _, server, _ = api
    npc_id = "captain_varra"
    player_id = "player_alden"
    campaign_id = "greyhaven_intro"

    server.save_turn(npc_id, player_id, "user", "Already stored.", campaign_id=campaign_id)
    server.save_player_npc_conversation_turns(
        npc_id,
        player_id,
        "Would duplicate if saved again.",
        {"spoken_text": "Hold the gate."},
        campaign_id=campaign_id,
        save_user=False,
    )

    history = server.load_recent_turns(npc_id, player_id, campaign_id=campaign_id)
    user_lines = [row["content"] for row in history if row["role"] == "user"]
    assistant_lines = [row["content"] for row in history if row["role"] == "assistant"]
    assert user_lines == ["Already stored."]
    assert "Hold the gate." in " ".join(assistant_lines)


def test_npc_turn_keeps_user_utterance_when_force_superseded(api, monkeypatch):
    """Confirm/Deny during a slow LLM must not drop the player line from history."""
    client, server, headers = api
    import turn_progress as tp

    npc_id = "captain_varra"
    player_id = "player_alden"
    campaign_id = "greyhaven_intro"
    request_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    player_line = "I swear I carry the captain's sealed writ."

    def _ask(_messages):
        active = tp.get_active_turn_progress(campaign_id, npc_id, player_id)
        assert active is not None
        assert active["request_id"] == request_id
        tp.register_turn_progress(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            route="/dm-ruling-auto-response",
            npc_id=npc_id,
            player_id=player_id,
            mode="text",
            campaign_id=campaign_id,
            force=True,
        )
        return "This wait reply should not be saved."

    monkeypatch.setattr(server, "ask_llm", _ask)

    turn = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": npc_id,
            "text": player_line,
            "scene": {"description": "Gate"},
            "history": [],
            "player_id": player_id,
            "campaign_id": campaign_id,
            "request_id": request_id,
            "include_audio": False,
        },
    )
    assert turn.status_code == 409
    body = turn.get_json()
    assert body.get("error") == "superseded"

    history = server.load_recent_turns(npc_id, player_id, campaign_id=campaign_id)
    user_lines = [row["content"] for row in history if row["role"] == "user"]
    assistant_lines = [row["content"] for row in history if row["role"] == "assistant"]
    assert player_line in user_lines
    assert not any("should not be saved" in (line or "").lower() for line in assistant_lines)
