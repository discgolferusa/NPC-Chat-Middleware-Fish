"""Unit + functional tests for short-clip Whisper hallucination detection."""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

from server import (
    _SHORT_CLIP_ALLOWLIST,
    _SHORT_CLIP_HALLUCINATIONS,
    _is_stt_hallucination,
    transcribe_audio,
)


def test_repeated_tokens_are_hallucinations():
    assert _is_stt_hallucination("No. No. No. No.")
    assert _is_stt_hallucination("Thank you. Thank you. Thank you.")


def test_normal_sentence_is_not_hallucination():
    assert not _is_stt_hallucination("Can you tell me about the harbor?")


def test_short_clip_you_is_hallucination():
    assert _is_stt_hallucination("You", duration_seconds=1.04, peak=1800)
    assert _is_stt_hallucination("Thank you", duration_seconds=1.2, peak=900)


def test_short_allowlisted_words_still_ok():
    assert not _is_stt_hallucination("Yes", duration_seconds=0.9, peak=8000)
    assert not _is_stt_hallucination("Wait", duration_seconds=1.0, peak=6000)
    # Previously ok/okay were also in the hallucination set, so the first check
    # rejected them before the allowlist could apply.
    assert not _is_stt_hallucination("Ok", duration_seconds=0.8, peak=900)
    assert not _is_stt_hallucination("Okay.", duration_seconds=1.1, peak=1200)
    assert not _is_stt_hallucination("OK", duration_seconds=1.0, peak=500)


def test_longer_you_sentence_not_rejected_by_short_clip_rule():
    assert not _is_stt_hallucination(
        "You there, what do you want from me?",
        duration_seconds=3.5,
        peak=10000,
    )


def test_allowlist_and_hallucinations_do_not_overlap():
    overlap = _SHORT_CLIP_ALLOWLIST & _SHORT_CLIP_HALLUCINATIONS
    assert not overlap, f"overlapping entries make allowlist dead: {sorted(overlap)}"


def _write_short_weak_wav(path: Path, *, duration_s: float = 1.0, peak: int = 900) -> None:
    rate = 22050
    n = max(1, int(rate * duration_s))
    # Alternating low samples so peak stays weak but not digitally silent.
    samples = [peak if i % 2 == 0 else -peak for i in range(n)]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{n}h", *samples))


class _FakeSttModel:
    def __init__(self, text: str):
        self._text = text
        self.calls = 0

    def transcribe(self, *args, **kwargs):
        self.calls += 1
        segment = SimpleNamespace(text=self._text)
        info = SimpleNamespace(duration=1.0)
        return [segment], info


def test_transcribe_audio_keeps_short_ok(tmp_path, monkeypatch):
    """Functional: allowlisted ok survives the full transcribe_audio filter path."""
    import server

    wav = tmp_path / "ok.wav"
    _write_short_weak_wav(wav)
    model = _FakeSttModel(" Okay.")
    monkeypatch.setattr(server, "STT_MODEL", model)
    monkeypatch.setattr(server, "_pad_wav_for_short_speech", lambda path: path)

    assert transcribe_audio(wav) == "Okay."
    assert model.calls >= 1


def test_transcribe_audio_rejects_short_you(tmp_path, monkeypatch):
    """Functional: bare You on a short/weak clip is cleared before callers see it."""
    import server

    wav = tmp_path / "you.wav"
    _write_short_weak_wav(wav)
    model = _FakeSttModel(" You")
    monkeypatch.setattr(server, "STT_MODEL", model)
    monkeypatch.setattr(server, "_pad_wav_for_short_speech", lambda path: path)

    assert transcribe_audio(wav) == ""
    assert model.calls >= 1


def test_npc_voice_turn_accepts_allowlisted_ok(api, tmp_path, monkeypatch):
    """HTTP path: short ok transcript is not rejected as empty speech."""
    import base64

    client, server, headers = api

    wav = tmp_path / "ok-upload.wav"
    _write_short_weak_wav(wav)
    model = _FakeSttModel("Ok")
    monkeypatch.setattr(server, "STT_MODEL", model)
    # Use the real transcribe_audio path (fixture stubs it by default).
    monkeypatch.setattr(server, "transcribe_audio", transcribe_audio)
    monkeypatch.setattr(server, "_pad_wav_for_short_speech", lambda path: path)

    response = client.post(
        "/npc-voice-turn",
        headers=headers,
        data={
            "npc_id": "captain_varra",
            "scene": "Barracks",
            "player_id": "player_alden",
            "audio": (io.BytesIO(wav.read_bytes()), "ok.wav"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    raw = response.headers["X-Player-Transcript"]
    padded = raw + ("=" * (-len(raw) % 4))
    transcript = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    assert transcript.lower() in {"ok", "okay"}
