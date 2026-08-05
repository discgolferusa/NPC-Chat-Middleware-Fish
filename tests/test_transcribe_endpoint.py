"""Functional coverage for POST /transcribe (Discord wake-word gating)."""

from __future__ import annotations

import io
import wave


def _silent_wav_bytes(duration_seconds: float = 0.5, sample_rate: int = 22050) -> bytes:
    frames = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def test_transcribe_endpoint_returns_transcript(api, monkeypatch):
    client, server, headers = api
    monkeypatch.setattr(server, "transcribe_audio", lambda path: "npcnarrator hello there")

    response = client.post(
        "/transcribe",
        headers=headers,
        data={"audio": (io.BytesIO(_silent_wav_bytes()), "clip.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["transcript"] == "npcnarrator hello there"


def test_transcribe_endpoint_requires_audio(api):
    client, _server, headers = api
    response = client.post("/transcribe", headers=headers)
    assert response.status_code == 400
    assert "audio" in (response.get_json() or {}).get("error", "").lower()
