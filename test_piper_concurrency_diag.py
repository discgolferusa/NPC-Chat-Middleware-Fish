"""Unit coverage for concurrency diag helpers (no real ONNX required)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import piper_pool
from scripts.diagnose_piper_concurrency import RequestTrace, _analyze, _intervals_overlap


def test_intervals_overlap():
    assert _intervals_overlap(0.0, 1.0, 0.5, 1.5) is True
    assert _intervals_overlap(0.0, 1.0, 1.0, 2.0) is False
    assert _intervals_overlap(0.0, 1.0, 2.0, 3.0) is False


def test_pool_allows_concurrent_synth_without_holding_pool_lock(monkeypatch, tmp_path):
    """Replicas checked out concurrently; _POOL_LOCK free during 'inference'."""
    piper_pool.clear_pool_for_tests()
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"onnx")
    (tmp_path / "voice.onnx.json").write_text(
        '{"num_speakers": 1, "speaker_id_map": {}}',
        encoding="utf-8",
    )

    class FakeVoice:
        def __init__(self, path: Path):
            self.model_path = path
            self.config = MagicMock(sample_rate=22050)
            self.session = object()  # distinct per instance
            self.in_synth = threading.Event()
            self.release_synth = threading.Event()

        def synthesize(self, text, syn_config=None):
            self.in_synth.set()
            assert self.release_synth.wait(timeout=2.0)
            chunk = MagicMock(
                sample_rate=22050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=b"\x00\x01" * 10,
            )
            yield chunk

    created: list[FakeVoice] = []

    def fake_create(path: Path):
        v = FakeVoice(path)
        created.append(v)
        return v

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "2")
    piper_pool.warm_voice_models([model_path], wait_for_ready=True, timeout_seconds=5.0)
    assert len(created) >= 2

    results: dict[str, object] = {}

    def run(name: str):
        out = tmp_path / f"{name}.wav"
        piper_pool.synthesize_wav_file(
            model_path,
            out,
            "hi",
            length_scale=1.0,
            noise_scale=0.667,
            noise_w=0.8,
            sentence_silence=0.0,
        )
        results[name] = out.is_file()

    t1 = threading.Thread(target=run, args=("a",))
    t2 = threading.Thread(target=run, args=("b",))
    t1.start()
    t2.start()
    assert created[0].in_synth.wait(timeout=1.0)
    assert created[1].in_synth.wait(timeout=1.0)
    # Both in synthesize concurrently → pool lock must be free.
    assert piper_pool._POOL_LOCK.acquire(timeout=0.2)
    piper_pool._POOL_LOCK.release()
    created[0].release_synth.set()
    created[1].release_synth.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    assert results.get("a") and results.get("b")


def test_analyze_detects_inference_overlap():
    traces = [
        RequestTrace(
            request_id="r1",
            thread_id=1,
            process_id=1,
            piper_instance_id="session:1",
            inference_start=1.0,
            inference_end=3.0,
        ),
        RequestTrace(
            request_id="r2",
            thread_id=2,
            process_id=1,
            piper_instance_id="session:2",
            inference_start=2.0,
            inference_end=4.0,
        ),
    ]
    analysis = _analyze(traces)
    assert analysis["inference_overlap_pair_count"] == 1
    assert analysis["verdict"] == "INFERENCE_WINDOWS_OVERLAP"
