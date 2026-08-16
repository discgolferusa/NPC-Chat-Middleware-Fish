"""Tests for early-return TTS streaming on /npc-voice-turn."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("faster_whisper")


def test_synthesize_returns_after_first_clip_and_finishes_in_background(monkeypatch, tmp_path):
    import server

    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        server,
        "get_narrator_tts_settings",
        lambda: {
            "resolved_voice_model": "narrator.onnx",
            "length_scale": 1.0,
            "noise_scale": 0.5,
            "noise_w": 0.5,
            "sentence_silence": 0.1,
            "speaker_id": 0,
            "pitch_semitones": 0.0,
            "fx_preset": "none",
            "denoise_nr": 0,
            "denoise_nf": -25,
        },
    )
    monkeypatch.setattr(
        server,
        "split_into_ordered_tts_segments",
        lambda text, pronunciation=None: [
            {"type": "npc", "text": "Hello there."},
            {"type": "npc", "text": "General Kenobi."},
        ],
    )

    calls: list[str] = []

    def fake_synth(text, voice, raw_path, **kwargs):
        calls.append(text)
        raw_path.write_bytes(b"RIFF" + b"\x00" * 40)

    def fake_convert(src, dest, **kwargs):
        dest.write_bytes(b"RIFF" + b"\x00" * 40)

    monkeypatch.setattr(server, "synthesize_speech", fake_synth)
    monkeypatch.setattr(server, "convert_wav_to_standard_format", fake_convert)

    done = {"ok": False}

    def on_complete(result, error):
        assert error is None
        assert len(result["clips"]) == 2
        done["ok"] = True

    result = server.synthesize_segmented_speech_clips(
        npc_text="ignored",
        npc_voice_model="npc.onnx",
        request_id="abc123stream",
        return_after_first_clip=True,
        on_stream_complete=on_complete,
    )

    assert result["pending"] is True
    assert len(result["clips"]) == 1
    assert result["clips"][0]["text"] == "Hello there."

    for _ in range(50):
        if done["ok"]:
            break
        time.sleep(0.05)

    assert done["ok"] is True
    assert calls == ["Hello there.", "General Kenobi."]



def test_synthesize_without_stream_flag_waits_for_all_clips(monkeypatch, tmp_path):
    import server

    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        server,
        "get_narrator_tts_settings",
        lambda: {
            "resolved_voice_model": "narrator.onnx",
            "length_scale": 1.0,
            "noise_scale": 0.5,
            "noise_w": 0.5,
            "sentence_silence": 0.1,
            "speaker_id": 0,
            "pitch_semitones": 0.0,
            "fx_preset": "none",
            "denoise_nr": 0,
            "denoise_nf": -25,
        },
    )
    monkeypatch.setattr(
        server,
        "split_into_ordered_tts_segments",
        lambda text, pronunciation=None: [
            {"type": "npc", "text": "One."},
            {"type": "npc", "text": "Two."},
        ],
    )
    monkeypatch.setattr(
        server,
        "synthesize_speech",
        lambda text, voice, raw_path, **kwargs: raw_path.write_bytes(b"RIFF" + b"\x00" * 40),
    )
    monkeypatch.setattr(
        server,
        "convert_wav_to_standard_format",
        lambda src, dest, **kwargs: dest.write_bytes(b"RIFF" + b"\x00" * 40),
    )

    result = server.synthesize_segmented_speech_clips(
        npc_text="ignored",
        npc_voice_model="npc.onnx",
        request_id="abc123full",
        return_after_first_clip=False,
    )

    assert result.get("pending") is False
    assert len(result["clips"]) == 2
