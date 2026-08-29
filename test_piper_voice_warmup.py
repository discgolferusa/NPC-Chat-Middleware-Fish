"""Functional tests for Piper warm-up on voice turns."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pytest.importorskip("faster_whisper")


def test_npc_voice_turn_schedules_piper_warm_before_stt(api, monkeypatch, tmp_path):
    client, server, headers = api
    scheduled: list[tuple[str, str | None, str]] = []

    def fake_schedule(npc_id, voice_override, *, request_id):
        scheduled.append((npc_id, voice_override, request_id))

    monkeypatch.setattr(server, "schedule_piper_warm_for_voice_turn", fake_schedule)

    call_order: list[str] = []

    def fake_transcribe(_path):
        call_order.append("stt")
        return "Hello captain."

    def fake_save(_file_storage, request_id: str):
        call_order.append("save")
        out = tmp_path / f"{request_id}.wav"
        out.write_bytes(b"\x00" * 1000)
        return out

    monkeypatch.setattr(server, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(server, "save_uploaded_audio", fake_save)

    response = client.post(
        "/npc-voice-turn",
        headers=headers,
        data={
            "npc_id": "captain_varra",
            "player_id": "player_1",
            "campaign_id": "camp_1",
            "audio": (io.BytesIO(b"fake-audio"), "clip.wav"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert len(scheduled) == 1
    assert scheduled[0][0] == "captain_varra"
    assert call_order == ["save", "stt"]
    steps = response.headers.get("X-Timing-Steps") or ""
    assert "schedule piper warm-up=" in steps


def test_resolve_voice_model_paths_for_turn_includes_npc_and_narrator(monkeypatch, tmp_path):
    import server

    npc_model = tmp_path / "npc-voice.onnx"
    narrator_model = tmp_path / "narrator-voice.onnx"
    npc_model.write_bytes(b"x")
    narrator_model.write_bytes(b"x")

    monkeypatch.setattr(
        server,
        "load_npc",
        lambda npc_id: {"voice_model": "npc-voice.onnx"},
    )
    monkeypatch.setattr(
        server,
        "resolve_voice_model_path",
        lambda voice_model: tmp_path / str(voice_model),
    )
    monkeypatch.setattr(
        server,
        "get_narrator_tts_settings",
        lambda: {"resolved_voice_model": "narrator-voice.onnx"},
    )

    paths = server.resolve_voice_model_paths_for_turn("captain_varra", None)
    assert paths == [npc_model, narrator_model]


def test_resolve_voice_model_paths_uses_default_when_npc_has_no_voice(monkeypatch, tmp_path):
    import server

    default_model = tmp_path / "default-voice.onnx"
    narrator_model = tmp_path / "narrator-voice.onnx"
    default_model.write_bytes(b"x")
    narrator_model.write_bytes(b"x")

    monkeypatch.setattr(server, "DEFAULT_VOICE_MODEL", default_model)
    monkeypatch.setattr(server, "load_npc", lambda _npc_id: {})
    monkeypatch.setattr(
        server,
        "resolve_voice_model_path",
        lambda voice_model: Path(voice_model),
    )
    monkeypatch.setattr(
        server,
        "get_narrator_tts_settings",
        lambda: {"resolved_voice_model": narrator_model},
    )

    paths = server.resolve_voice_model_paths_for_turn("unvoiced_npc", None)

    assert paths == [default_model, narrator_model]
