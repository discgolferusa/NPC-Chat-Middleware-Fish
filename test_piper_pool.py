"""Unit tests for in-process Piper voice pool warm-up."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import piper_pool


class _FakeVoice:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.config = MagicMock(sample_rate=22050)

    def synthesize(self, text, syn_config=None):
        chunk = MagicMock(
            sample_rate=22050,
            sample_width=2,
            sample_channels=1,
            audio_int16_bytes=b"\x00\x01" * 100,
        )
        yield chunk


def test_warm_voice_models_loads_in_background(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = tmp_path / "npc.onnx"
    model_path.write_bytes(b"onnx")

    load_started = threading.Event()
    load_done = threading.Event()

    def fake_create(path: Path):
        load_started.set()
        time.sleep(0.05)
        load_done.set()
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)

    piper_pool.warm_voice_models([model_path], request_id="req-warm")

    assert load_started.wait(timeout=1.0)
    assert load_done.wait(timeout=1.0)

    with piper_pool._POOL_LOCK:
        assert str(model_path) in piper_pool._SLOTS
        assert piper_pool._SLOTS[str(model_path)].voice is not None


def test_warm_voice_models_is_idempotent(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = tmp_path / "npc.onnx"
    model_path.write_bytes(b"onnx")
    calls = {"count": 0}

    def fake_create(path: Path):
        calls["count"] += 1
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)

    piper_pool.warm_voice_models([model_path, model_path], request_id="req1")
    time.sleep(0.05)
    piper_pool.warm_voice_models([model_path], request_id="req2")
    time.sleep(0.05)

    assert calls["count"] == 1


def test_synthesize_wav_file_uses_pooled_voice(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = tmp_path / "npc.onnx"
    model_path.write_bytes(b"onnx")
    output_path = tmp_path / "out.wav"

    fake_voice = _FakeVoice(model_path)
    monkeypatch.setattr(
        piper_pool,
        "_get_voice_and_lock",
        lambda path: (fake_voice, threading.Lock()),
    )

    piper_pool.synthesize_wav_file(
        model_path,
        output_path,
        "Hello there.",
        length_scale=1.0,
        noise_scale=0.667,
        noise_w=0.8,
        sentence_silence=0.1,
        speaker_id=3,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_get_voice_waits_for_warm_thread(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = tmp_path / "wait.onnx"
    model_path.write_bytes(b"onnx")

    started = threading.Event()

    def slow_create(path: Path):
        started.set()
        time.sleep(0.08)
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", slow_create)
    piper_pool.warm_voice_models([model_path], request_id="req-wait")
    assert started.wait(timeout=1.0)

    voice, _lock = piper_pool._get_voice_and_lock(model_path)
    assert voice.model_path == model_path


def test_cold_path_load_does_not_block_other_models(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    slow_path = tmp_path / "slow.onnx"
    fast_path = tmp_path / "fast.onnx"
    slow_path.write_bytes(b"onnx")
    fast_path.write_bytes(b"onnx")

    load_started = threading.Event()
    release_slow = threading.Event()

    def fake_create(path: Path):
        if path == slow_path:
            load_started.set()
            assert release_slow.wait(timeout=1.0)
            return _FakeVoice(path)
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)

    slow_result: dict[str, Any] = {}
    fast_result: dict[str, Any] = {}

    def load_slow():
        voice, lock = piper_pool._get_voice_and_lock(slow_path)
        slow_result["voice"] = voice
        slow_result["lock"] = lock

    def load_fast():
        load_started.wait(timeout=1.0)
        voice, lock = piper_pool._get_voice_and_lock(fast_path)
        fast_result["voice"] = voice
        fast_result["lock"] = lock

    slow_thread = threading.Thread(target=load_slow)
    slow_thread.start()
    assert load_started.wait(timeout=1.0)

    fast_thread = threading.Thread(target=load_fast)
    fast_start = time.perf_counter()
    fast_thread.start()
    fast_thread.join(timeout=1.0)
    fast_elapsed = time.perf_counter() - fast_start

    assert fast_thread.is_alive() is False
    assert "voice" in fast_result
    assert fast_elapsed < 0.25

    release_slow.set()
    slow_thread.join(timeout=1.0)
    assert "voice" in slow_result
