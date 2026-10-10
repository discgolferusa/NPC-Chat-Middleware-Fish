"""Unit + functional tests for PIPER_ONNX_INTRA_OP_THREADS and ORT bench helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import piper_pool


def test_onnx_intra_op_threads_default_zero(monkeypatch):
    monkeypatch.delenv("PIPER_ONNX_INTRA_OP_THREADS", raising=False)
    assert piper_pool.onnx_intra_op_threads() == 0


def test_onnx_intra_op_threads_zero_string(monkeypatch):
    monkeypatch.setenv("PIPER_ONNX_INTRA_OP_THREADS", "0")
    assert piper_pool.onnx_intra_op_threads() == 0


def test_onnx_intra_op_threads_parses_positive(monkeypatch):
    monkeypatch.setenv("PIPER_ONNX_INTRA_OP_THREADS", "2")
    assert piper_pool.onnx_intra_op_threads() == 2


def test_onnx_intra_op_threads_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("PIPER_ONNX_INTRA_OP_THREADS", "nope")
    assert piper_pool.onnx_intra_op_threads() == 0


def test_create_piper_voice_default_uses_stock_load(monkeypatch, tmp_path):
    model = tmp_path / "v.onnx"
    model.write_bytes(b"onnx")
    called: dict = {}

    class FakeVoice:
        pass

    def fake_load(path, use_cuda=False, **kwargs):
        called["load"] = (Path(path), use_cuda)
        return FakeVoice()

    monkeypatch.delenv("PIPER_ONNX_INTRA_OP_THREADS", raising=False)
    monkeypatch.delenv("PIPER_USE_CUDA", raising=False)

    import piper

    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(fake_load))

    voice = piper_pool._create_piper_voice(model)
    assert isinstance(voice, FakeVoice)
    assert called["load"][0] == model
    assert called["load"][1] is False


def test_create_piper_voice_custom_threads_builds_session(monkeypatch, tmp_path):
    model = tmp_path / "v.onnx"
    model.write_bytes(b"onnx")
    (tmp_path / "v.onnx.json").write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": "en-us"},
                "inference": {},
                "phoneme_type": "espeak",
                "phoneme_id_map": {"_": [0]},
                "num_speakers": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PIPER_ONNX_INTRA_OP_THREADS", "2")
    monkeypatch.delenv("PIPER_USE_CUDA", raising=False)

    captured: dict = {}

    class Opts:
        def __init__(self):
            self.intra_op_num_threads = 0

    class FakeSession:
        def __init__(self, model_path, sess_options=None, providers=None):
            captured["model_path"] = model_path
            captured["sess_options"] = sess_options
            captured["providers"] = providers

    class FakeConfig:
        @staticmethod
        def from_dict(d):
            return SimpleNamespace(raw=d)

    class FakeVoice:
        def __init__(self, **kwargs):
            captured["voice_kwargs"] = kwargs

    import onnxruntime
    import piper
    import piper.config as piper_config

    monkeypatch.setattr(onnxruntime, "SessionOptions", Opts)
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    monkeypatch.setattr(piper, "PiperVoice", FakeVoice)
    monkeypatch.setattr(piper_config, "PiperConfig", FakeConfig)

    # Stock load must NOT be used when threads > 0
    def boom(*_a, **_k):
        raise AssertionError("PiperVoice.load should not be called when threads>0")

    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(boom), raising=False)

    voice = piper_pool._create_piper_voice(model)
    assert isinstance(voice, FakeVoice)
    assert isinstance(captured["sess_options"], Opts)
    assert captured["sess_options"].intra_op_num_threads == 2
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert isinstance(captured["voice_kwargs"]["session"], FakeSession)


def test_ort_bench_pick_best_helpers():
    from scripts.benchmark_piper_ort_threads import pick_best_intra_op

    cells = [
        {
            "intra_op_threads": 0,
            "dialogue_class": "medium",
            "concurrency": 4,
            "wall_ms": {"p50": 900.0, "p95": 1200.0},
            "throughput_rps": 2.0,
        },
        {
            "intra_op_threads": 1,
            "dialogue_class": "medium",
            "concurrency": 4,
            "wall_ms": {"p50": 600.0, "p95": 800.0},
            "throughput_rps": 3.5,
        },
        {
            "intra_op_threads": 2,
            "dialogue_class": "medium",
            "concurrency": 4,
            "wall_ms": {"p50": 650.0, "p95": 850.0},
            "throughput_rps": 3.2,
        },
    ]
    best = pick_best_intra_op(cells, dialogue_class="medium", concurrency=4)
    assert best["intra_op_threads"] == 1
    assert best["criterion"] == "lowest_wall_p95_then_p50"


def test_ort_bench_variability_summary():
    import statistics

    from scripts.benchmark_piper_ort_threads import summarize_repeat_variability

    repeats = [
        {"wall_ms": {"p50": 100.0, "p95": 200.0}, "throughput_rps": 1.0},
        {"wall_ms": {"p50": 120.0, "p95": 220.0}, "throughput_rps": 1.2},
        {"wall_ms": {"p50": 110.0, "p95": 210.0}, "throughput_rps": 1.1},
    ]
    v = summarize_repeat_variability(repeats)
    assert v["wall_p50_mean"] == 110.0
    assert v["wall_p50_stdev"] == statistics.stdev([100.0, 120.0, 110.0])
    assert v["n_repeats"] == 3


def test_rebuild_pool_sets_env_and_clears(monkeypatch, tmp_path):
    """Functional: each thread config rebuilds via clear + warm (new sessions)."""
    from scripts import benchmark_piper_ort_threads as bench

    model = tmp_path / "libritts.onnx"
    model.write_bytes(b"onnx")
    (tmp_path / "libritts.onnx.json").write_text('{"num_speakers": 1}', encoding="utf-8")

    calls: list[str] = []

    def fake_clear():
        calls.append("clear")

    def fake_warm(paths, **kwargs):
        calls.append(f"warm:{piper_pool.onnx_intra_op_threads()}")
        return {"ok": True}

    monkeypatch.setattr(piper_pool, "clear_pool_for_tests", fake_clear)
    monkeypatch.setattr(piper_pool, "warm_voice_models", fake_warm)
    monkeypatch.setattr(piper_pool, "pool_status", lambda: {})
    monkeypatch.setattr(bench, "_synth_one", lambda **kwargs: None)
    monkeypatch.setattr(bench, "_rss_bytes", lambda: 1)

    info = bench._rebuild_pool(model, slots=8, intra=2)
    assert calls == ["clear", "warm:2"]
    assert info["effective_onnx_intra_op_threads"] == 2
    assert os.environ.get("PIPER_ONNX_INTRA_OP_THREADS") == "2"
