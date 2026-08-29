"""Context-loading concurrency and fresh-DM prompt regression tests."""

from __future__ import annotations

import threading


def _stub_context_loaders(server, monkeypatch, *, dm_loader):
    monkeypatch.setattr(server, "load_npc", lambda _npc_id: {"location_id": "gate"})
    monkeypatch.setattr(server, "load_memories", lambda *_args, **_kwargs: ["memory"])
    monkeypatch.setattr(server, "load_recent_turns", lambda *_args, **_kwargs: ["turn"])
    monkeypatch.setattr(server, "get_dm_context", dm_loader)
    monkeypatch.setattr(server, "load_location", lambda _location_id: {"name": "Gate"})
    monkeypatch.setattr(server, "build_scene_context", lambda _location_id: {"time": "night"})


def test_independent_context_reads_overlap(api, monkeypatch):
    _client, server, _headers = api
    barrier = threading.Barrier(4, timeout=2.0)

    def npc_loader(_npc_id):
        barrier.wait()
        return {"location_id": "gate"}

    def memory_loader(*_args, **_kwargs):
        barrier.wait()
        return ["memory"]

    def turns_loader(*_args, **_kwargs):
        barrier.wait()
        return ["turn"]

    def dm_loader(*_args, **_kwargs):
        barrier.wait()
        return {"messages": [], "pending_checks": [], "rulings": []}

    monkeypatch.setattr(server, "load_npc", npc_loader)
    monkeypatch.setattr(server, "load_memories", memory_loader)
    monkeypatch.setattr(server, "load_recent_turns", turns_loader)
    monkeypatch.setattr(server, "get_dm_context", dm_loader)
    monkeypatch.setattr(server, "load_location", lambda _location_id: {"name": "Gate"})
    monkeypatch.setattr(server, "build_scene_context", lambda _location_id: {"time": "night"})

    context = server.get_npc_context(
        "captain_varra",
        "player_alden",
        party_context={"party_id": "party", "campaign_id": "campaign"},
    )

    assert context["npc"]["location_id"] == "gate"
    assert context["memories"] == ["memory"]
    assert context["recent_turns"] == ["turn"]
    assert context["dm_context"]["messages"] == []


def test_dm_context_is_optional_but_remains_enabled_by_default(api, monkeypatch):
    _client, server, _headers = api
    calls = {"count": 0}

    def dm_loader(*_args, **_kwargs):
        calls["count"] += 1
        return {"source": "fresh"}

    _stub_context_loaders(server, monkeypatch, dm_loader=dm_loader)
    party_context = {"party_id": "party", "campaign_id": "campaign"}

    skipped = server.get_npc_context(
        "captain_varra",
        "player_alden",
        party_context=party_context,
        include_dm_context=False,
    )
    defaulted = server.get_npc_context(
        "captain_varra",
        "player_alden",
        party_context=party_context,
    )

    assert skipped["dm_context"] == {}
    assert defaulted["dm_context"] == {"source": "fresh"}
    assert calls["count"] == 1


def test_npc_turn_loads_dm_once_after_processing_the_new_claim(api, monkeypatch):
    client, server, headers = api
    events: list[str] = []
    fresh_dm = {
        "campaign_id": "greyhaven_intro",
        "messages": [{"content": "fresh DM note", "detected_verdict": "unclear"}],
        "pending_checks": [],
        "rulings": [],
    }

    def process_player(*_args, **_kwargs):
        events.append("process_player_message")
        return None

    def load_dm(*_args, **_kwargs):
        events.append("get_dm_context")
        return fresh_dm

    def build_messages(*_args, **kwargs):
        events.append("build_prompt")
        assert kwargs["dm_context"] is fresh_dm
        return [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Player prompt"},
        ]

    monkeypatch.setattr(server, "process_player_message", process_player)
    monkeypatch.setattr(server, "get_dm_context", load_dm)
    monkeypatch.setattr(server, "build_npc_llm_messages", build_messages)
    monkeypatch.setattr(
        server,
        "generate_npc_reply_text",
        lambda *_args, **_kwargs: "The gate remains closed.",
    )

    response = client.post(
        "/npc-turn",
        headers=headers,
        json={
            "npc_id": "captain_varra",
            "player_id": "player_alden",
            "text": "I have permission to enter.",
            "include_audio": False,
            "request_id": "contextfreshdm0000000000000000001",
        },
    )

    assert response.status_code == 200
    assert events == ["process_player_message", "get_dm_context", "build_prompt"]
