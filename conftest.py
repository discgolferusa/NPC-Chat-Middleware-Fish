"""Shared Flask test client for game API functional tests."""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("flask")


@pytest.fixture()
def api(monkeypatch, tmp_path):
    """Authenticated Flask client with fully isolated on-disk side effects.

    Redirects MEMORY_DB (including module-level init during reload), UPLOAD_DIR,
    OUTPUT_DIR, and session/scene runtime config files into ``tmp_path`` so tests
    cannot overwrite a developer's local defaults or leave upload artifacts.
    """
    monkeypatch.setenv("GAME_API_KEY", "test-secret-key")
    monkeypatch.setenv("REQUIRE_GAME_API_KEY", "false")
    monkeypatch.delenv("VOICES_DIR", raising=False)

    mem_db = (tmp_path / "npc_memory.db").resolve()
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    session_cfg = tmp_path / "session_runtime_config.json"
    scene_cfg = tmp_path / "scene_runtime_config.json"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    import dm_channel
    import party_campaign
    import scene_state
    import server
    import turn_progress

    repo_memory_db = (Path(server.__file__).resolve().parent / "npc_memory.db").resolve()
    real_connect = sqlite3.connect

    def _connect(database, *args, **kwargs):
        """Keep reload-time init_memory_db() off the developer npc_memory.db."""
        try:
            db_path = Path(database).resolve()
        except (TypeError, ValueError, OSError):
            return real_connect(database, *args, **kwargs)
        if db_path == repo_memory_db:
            return real_connect(str(mem_db), *args, **kwargs)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _connect)

    importlib.reload(dm_channel)
    importlib.reload(server)

    monkeypatch.setattr(server, "MEMORY_DB", mem_db)
    monkeypatch.setattr(dm_channel, "MEMORY_DB", mem_db)
    monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(server, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(server, "GAME_API_KEY", "test-secret-key")
    monkeypatch.setattr(server, "REQUIRE_GAME_API_KEY", False)
    monkeypatch.setattr(server, "AUTO_MEMORY_EXTRACTION_ENABLED", False)

    monkeypatch.setattr(scene_state, "SCENE_RUNTIME_CONFIG_FILE", scene_cfg)
    monkeypatch.setattr(party_campaign, "SESSION_RUNTIME_CONFIG_FILE", session_cfg)

    # Fresh in-memory session/scene state (do not load developer runtime JSON).
    with scene_state._STATE_LOCK:
        scene_state._SCENE_SESSIONS.clear()
    scene_state._FALLBACK_TIME_OF_DAY = ""
    scene_state._FALLBACK_SUB_LOCATION = None

    with party_campaign._STATE_LOCK:
        party_campaign._PLAY_SESSIONS.clear()
    # Deterministic local/dev fallbacks (not loaded from the developer's session JSON).
    party_campaign._FALLBACK_CAMPAIGN_ID = "greyhaven_intro"
    party_campaign._FALLBACK_PARTY_ID = "default_party"
    party_campaign._FALLBACK_PLAYER_ID = "player_alden"

    with turn_progress._PROGRESS_LOCK:
        turn_progress._PROGRESS.clear()

    monkeypatch.setattr(server, "ask_llm", lambda messages: "The watch holds firm tonight.")
    monkeypatch.setattr(server, "transcribe_audio", lambda path: "Hello captain.")

    def _fake_tts(**kwargs):
        request_id = kwargs.get("request_id") or "test"
        speech_path = server.turn_audio_path(request_id, 0)
        speech_path.write_bytes(
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        dialogue = kwargs.get("npc_text") or "The watch holds firm tonight."
        return {
            "clips": [
                {
                    "index": 0,
                    "type": "speech",
                    "text": dialogue,
                    "url": server.turn_audio_url(request_id, 0),
                    "path": str(speech_path),
                }
            ],
            "narration_text": "",
            "dialogue_text": dialogue,
            "speech_path": speech_path,
            "narration_path": None,
        }

    monkeypatch.setattr(server, "synthesize_segmented_speech_clips", _fake_tts)
    monkeypatch.setattr(server, "refresh_voice_inventory", lambda force=False: [
        {"name": "en_US-lessac-medium.onnx", "path": "en_US-lessac-medium.onnx"},
    ])

    monkeypatch.setattr(
        server,
        "analyze_campaign_document",
        lambda *a, **k: {"campaign": {"id": "test_campaign", "name": "Test"}, "locations": [], "npcs": []},
    )
    monkeypatch.setattr(
        server,
        "outline_campaign_document",
        lambda *a, **k: {"campaign": {"id": "test_campaign", "name": "Test"}, "locations": [], "npcs": []},
    )
    monkeypatch.setattr(
        server,
        "detail_location_batch",
        lambda *a, **k: {"locations": [{"id": "loc_1", "name": "Loc One"}]},
    )
    monkeypatch.setattr(
        server,
        "detail_npc_batch",
        lambda *a, **k: {"npcs": [{"id": "npc_1", "name": "Npc One"}]},
    )

    dm_channel.init_dm_tables()
    server.init_memory_db()
    server.app.config["TESTING"] = True

    headers = {"Authorization": "Bearer test-secret-key"}
    with server.app.test_client() as client:
        yield client, server, headers
