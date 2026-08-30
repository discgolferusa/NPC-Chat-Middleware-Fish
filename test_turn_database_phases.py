"""Regression tests for the short pre/post-LLM database phases."""

from __future__ import annotations

import io

import pytest


@pytest.mark.parametrize("failure_stage", ["npc", "location"])
def test_static_context_failure_persists_no_pre_llm_writes(
    api, monkeypatch, failure_stage
):
    _client, server, _headers = api

    def fail_static_load(*_args, **_kwargs):
        raise LookupError(f"missing {failure_stage}")

    if failure_stage == "npc":
        monkeypatch.setattr(server, "load_npc", fail_static_load)
    else:
        monkeypatch.setattr(
            server,
            "load_npc",
            lambda _npc_id: {"id": "missing_location_npc", "location_id": "missing"},
        )
        monkeypatch.setattr(server, "load_location", fail_static_load)
        monkeypatch.setattr(server, "build_scene_context", lambda _location_id: {})

    with pytest.raises(LookupError, match=f"missing {failure_stage}"):
        server.load_turn_pre_llm_context(
            "missing_npc",
            "player_alden",
            player_text="We have a letter from the mayor allowing us to enter.",
            party_context={"party_id": "default_party", "campaign_id": "greyhaven_intro"},
            campaign_id="greyhaven_intro",
        )

    with server.get_db() as conn:
        turn_count = conn.execute(
            "SELECT COUNT(*) AS count FROM conversation_turns"
        ).fetchone()["count"]
        pending_count = conn.execute(
            "SELECT COUNT(*) AS count FROM dm_pending_checks"
        ).fetchone()["count"]
    assert turn_count == 0
    assert pending_count == 0


def test_pre_llm_database_phase_starts_only_after_static_context(api, monkeypatch):
    _client, server, _headers = api
    events: list[str] = []

    def static_context(*_args, **_kwargs):
        events.append("static_complete")
        return {"npc": {}, "location": {}, "scene_context": {}}

    def database_phase(*_args, **_kwargs):
        events.append("database_phase")
        return {
            "memories": [],
            "recent_turns": [],
            "player_claim": None,
            "dm_context": {},
        }

    monkeypatch.setattr(server, "get_npc_context", static_context)
    monkeypatch.setattr(server, "prepare_turn_database_phase", database_phase)

    server.load_turn_pre_llm_context(
        "captain_varra",
        "player_alden",
        player_text="Hello.",
        party_context={"party_id": "default_party", "campaign_id": "greyhaven_intro"},
        campaign_id="greyhaven_intro",
    )

    assert events == ["static_complete", "database_phase"]


def test_pre_llm_phase_uses_one_connection_and_saves_player(api, monkeypatch):
    _client, server, _headers = api
    original_get_db = server.get_db
    calls = {"count": 0}

    def counted_get_db():
        calls["count"] += 1
        return original_get_db()

    monkeypatch.setattr(server, "get_db", counted_get_db)

    result = server.prepare_turn_database_phase(
        "captain_varra",
        "player_alden",
        "We have a letter from the mayor allowing us to enter.",
        campaign_id="greyhaven_intro",
    )

    assert calls["count"] == 1
    assert result["player_claim"] is not None
    pending = result["dm_context"]["pending_checks"]
    assert len(pending) == 1
    assert pending[0]["id"] == result["player_claim"]["pending_check_id"]

    with original_get_db() as conn:
        turns = conn.execute(
            """
            SELECT role, content FROM conversation_turns
            WHERE campaign_id = ? AND npc_id = ? AND player_id = ?
            ORDER BY id
            """,
            ("greyhaven_intro", "captain_varra", "player_alden"),
        ).fetchall()
    assert [(row["role"], row["content"]) for row in turns] == [
        ("user", "We have a letter from the mayor allowing us to enter.")
    ]


def test_post_llm_phase_uses_one_connection_and_returns_pending_snapshot(
    api, monkeypatch
):
    _client, server, _headers = api
    import dm_channel

    original_server_get_db = server.get_db
    server_calls = {"count": 0}
    unexpected_dm_connections = {"count": 0}

    def counted_server_get_db():
        server_calls["count"] += 1
        return original_server_get_db()

    def counted_dm_get_db():
        unexpected_dm_connections["count"] += 1
        return original_server_get_db()

    monkeypatch.setattr(server, "get_db", counted_server_get_db)
    monkeypatch.setattr(dm_channel, "get_db", counted_dm_get_db)

    result = server.finalize_turn_database_phase(
        "captain_varra",
        "player_alden",
        "Open the gate.",
        "Not yet. [DM CHECK: the gate may be opened]",
        campaign_id="greyhaven_intro",
        npc={"name": "Captain Varra"},
    )

    assert server_calls["count"] == 1
    assert unexpected_dm_connections["count"] == 0
    assert result["pending_count"] == 1
    assert len(result["pending_items"]) == 1
    assert result["npc_response"]["pending_check_id"] == result["pending_items"][0]["id"]

    with original_server_get_db() as conn:
        assistant = conn.execute(
            """
            SELECT content FROM conversation_turns
            WHERE campaign_id = ? AND npc_id = ? AND player_id = ? AND role = 'assistant'
            """,
            ("greyhaven_intro", "captain_varra", "player_alden"),
        ).fetchone()
    assert assistant["content"] == "Not yet."


def test_complete_log_reuses_supplied_pending_count(api, monkeypatch, capsys):
    _client, _server, _headers = api
    import dm_channel

    def fail_if_queried(*_args, **_kwargs):
        raise AssertionError("logging must not reload pending checks")

    monkeypatch.setattr(dm_channel, "load_pending_checks", fail_if_queried)
    dm_channel.log_dm_turn(
        request_id="req-1",
        route="/npc-turn",
        npc_id="captain_varra",
        player_id="player_alden",
        stage="complete",
        campaign_id="greyhaven_intro",
        pending_count=3,
    )

    assert "pending_count=3" in capsys.readouterr().out


def test_locked_pending_load_adds_for_update_only_for_mysql():
    import dm_channel

    statements: list[str] = []

    class _Config:
        is_mysql = True

    class _Result:
        def fetchall(self):
            return []

    class _Connection:
        config = _Config()

        def execute(self, sql, _params=None):
            statements.append(sql)
            return _Result()

    rows = dm_channel.load_pending_checks(
        "captain_varra",
        "player_alden",
        campaign_id="greyhaven_intro",
        connection=_Connection(),  # type: ignore[arg-type]
        for_update=True,
    )

    assert rows == []
    assert statements[0].strip().endswith("FOR UPDATE")


def test_voice_route_does_not_reload_pending_after_piper(api, monkeypatch, tmp_path):
    client, server, headers = api
    original_load_pending = server.load_pending_checks
    original_tts = server.synthesize_segmented_speech_clips
    events: list[str] = []

    def tracked_pending(*args, **kwargs):
        events.append("pending")
        return original_load_pending(*args, **kwargs)

    def tracked_tts(**kwargs):
        events.append("piper")
        return original_tts(**kwargs)

    def fake_save(_file_storage, request_id):
        path = tmp_path / f"{request_id}.wav"
        path.write_bytes(b"fake-audio")
        return path

    monkeypatch.setattr(server, "load_pending_checks", tracked_pending)
    monkeypatch.setattr(server, "synthesize_segmented_speech_clips", tracked_tts)
    monkeypatch.setattr(server, "save_uploaded_audio", fake_save)
    monkeypatch.setattr(server, "schedule_piper_warm_for_voice_turn", lambda *_a, **_k: None)

    response = client.post(
        "/npc-voice-turn",
        headers=headers,
        data={
            "npc_id": "captain_varra",
            "player_id": "player_alden",
            "campaign_id": "greyhaven_intro",
            "audio": (io.BytesIO(b"fake-audio"), "clip.wav"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert events.count("pending") == 2  # pre-LLM snapshot + post-LLM snapshot
    assert events[-1] == "piper"
