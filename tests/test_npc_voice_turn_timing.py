"""Timing headers on /npc-voice-turn failure paths."""

from __future__ import annotations

import io

import pytest

pytest.importorskip("faster_whisper")


def test_npc_voice_turn_no_speech_400_includes_timing_headers(api, monkeypatch, tmp_path):
    client, server, headers = api

    def fake_save(_file_storage, request_id: str):
        out = tmp_path / f"{request_id}.wav"
        out.write_bytes(b"\x00" * 1000)
        return out

    monkeypatch.setattr(server, "save_uploaded_audio", fake_save)
    monkeypatch.setattr(server, "transcribe_audio", lambda _path: "")

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

    assert response.status_code == 400
    assert response.headers.get("X-Timing-Total-Ms") is not None
    steps = response.headers.get("X-Timing-Steps") or ""
    assert "speech-to-text=" in steps
