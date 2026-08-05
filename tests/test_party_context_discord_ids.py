"""Unit coverage: party_members keep Discord identity for API/session consumers."""

from __future__ import annotations

import party_campaign as pc


def test_build_party_context_preserves_discord_user_id(monkeypatch):
    member = {
        "player_id": "player_alden",
        "party_id": "heroes",
        "campaign_id": "greyhaven_intro",
        "name": "Alden",
        "race": "Human",
        "class": "Fighter",
        "discord_user_id": "424242424242",
        "discord_handle": "alden#1",
    }

    monkeypatch.setattr(pc, "_current_play_ids", lambda: {
        "player_id": "player_alden",
        "party_id": "heroes",
        "campaign_id": "greyhaven_intro",
    })
    monkeypatch.setattr(pc, "resolve_member", lambda player_id, party_id=None: dict(member))
    monkeypatch.setattr(
        pc,
        "load_party",
        lambda party_id: {"party_id": party_id, "members": [dict(member)], "campaign_id": "greyhaven_intro"},
    )
    monkeypatch.setattr(pc, "load_campaign", lambda campaign_id: {"campaign_id": campaign_id, "name": "Greyhaven"})
    monkeypatch.setattr(pc, "get_character_options", lambda **kwargs: [])

    ctx = pc.build_party_context(player_id="player_alden", party_id="heroes")
    assert ctx["party_members"][0]["discord_user_id"] == "424242424242"
    assert ctx["party_members"][0]["discord_handle"] == "alden#1"
    assert ctx["speaking_member"]["discord_user_id"] == "424242424242"

    prompt = pc.format_party_context_for_prompt(ctx)
    assert "424242424242" not in prompt
    assert "discord" not in prompt.lower()
    assert "Alden" in prompt
