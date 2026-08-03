"""Shared Flask test client for game API functional tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("flask")


@pytest.fixture()
def api(monkeypatch, tmp_path):
    """Authenticated Flask client with isolated DBs and disabled live LLM/TTS/STT."""
    monkeypatch.setenv("GAME_API_KEY", "test-secret-key")
    monkeypatch.setenv("REQUIRE_GAME_API_KEY", "false")
    monkeypatch.delenv("VOICES_DIR", raising=False)

    import dm_channel
    import server
    import turn_progress

    importlib.reload(dm_channel)
    importlib.reload(server)

    mem_db = tmp_path / "npc_memory.db"
    monkeypatch.setattr(server, "MEMORY_DB", mem_db)
    monkeypatch.setattr(dm_channel, "MEMORY_DB", mem_db)
    monkeypatch.setattr(server, "GAME_API_KEY", "test-secret-key")
    monkeypatch.setattr(server, "REQUIRE_GAME_API_KEY", False)
    monkeypatch.setattr(server, "AUTO_MEMORY_EXTRACTION_ENABLED", False)
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path / "output")
    server.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Isolate in-memory scene/progress state between tests.
    with turn_progress._PROGRESS_LOCK:
        turn_progress._PROGRESS.clear()

    monkeypatch.setattr(server, "ask_llm", lambda messages: "The watch holds firm tonight.")
    monkeypatch.setattr(server, "transcribe_audio", lambda path: "Hello captain.")

    def _fake_tts(**kwargs):
        request_id = kwargs.get("request_id") or "test"
        speech_path = server.turn_audio_path(request_id, 0)
        # Minimal RIFF/WAVE header so send_file has a real path.
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

    # Content-import HTTP routes: mock LLM-backed analyzers.
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
