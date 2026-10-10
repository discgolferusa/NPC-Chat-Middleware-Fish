"""Functional: /voices/preview exposes timing + Piper pool instrumentation headers."""

from __future__ import annotations

from pathlib import Path


def test_voices_preview_attaches_timing_and_piper_headers(api, monkeypatch, tmp_path):
    client, server, headers = api

    wav = tmp_path / "preview.wav"
    wav.write_bytes(b"RIFF....WAVE")

    def fake_synth(text, voice_model, output_path, **kwargs):
        Path(output_path).write_bytes(b"RIFF....WAVE")

    def fake_convert(input_path, output_path, **kwargs):
        Path(output_path).write_bytes(b"RIFF....WAVEconverted")

    monkeypatch.setattr(server, "resolve_voice_model_path", lambda p: tmp_path / "v.onnx")
    monkeypatch.setattr(server, "synthesize_speech", fake_synth)
    monkeypatch.setattr(server, "convert_wav_to_standard_format", fake_convert)
    monkeypatch.setattr(
        server.piper_pool,
        "last_call_metrics",
        lambda: {"acquire_wait_ms": 12, "synth_ms": 34},
    )
    monkeypatch.setattr(
        server.piper_pool,
        "pool_runtime_metrics",
        lambda: {"active_slots": 1, "waiting_requests": 0, "slot_budget": 10, "models": {}},
    )

    response = client.post(
        "/voices/preview",
        headers=headers,
        json={"voice_path": str(tmp_path / "v.onnx"), "text": "Hello preview."},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Timing-Total-Ms") is not None
    assert "preview piper synthesize" in (response.headers.get("X-Timing-Steps") or "")
    assert response.headers.get("X-Piper-Acquire-Wait-Ms") == "12"
    assert response.headers.get("X-Piper-Synth-Ms") == "34"
    assert response.headers.get("X-Piper-Active-Slots") == "1"
    assert response.headers.get("X-Piper-Waiting-Requests") == "0"


def test_piper_pool_metrics_endpoint(api):
    client, _server, headers = api
    response = client.get("/piper-pool/metrics", headers=headers)
    assert response.status_code == 200
    body = response.get_json()
    assert "active_slots" in body
    assert "waiting_requests" in body
    assert "models" in body
