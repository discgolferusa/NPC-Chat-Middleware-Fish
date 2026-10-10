"""Unit + functional tests for first-chunk reserved pool acquires (P0)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import piper_pool


def test_first_chunk_reserved_default(monkeypatch):
    monkeypatch.delenv("PIPER_FIRST_CHUNK_RESERVED", raising=False)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "10")
    assert piper_pool.first_chunk_reserved_slots() == 2


def test_first_chunk_reserved_zero_disables(monkeypatch):
    monkeypatch.setenv("PIPER_FIRST_CHUNK_RESERVED", "0")
    monkeypatch.setenv("PIPER_POOL_SLOTS", "10")
    assert piper_pool.first_chunk_reserved_slots() == 0


def test_first_chunk_reserved_capped_by_budget(monkeypatch):
    monkeypatch.setenv("PIPER_FIRST_CHUNK_RESERVED", "9")
    monkeypatch.setenv("PIPER_POOL_SLOTS", "4")
    assert piper_pool.first_chunk_reserved_slots() == 3  # budget - 1


def test_followup_yields_while_first_chunk_waits(monkeypatch, tmp_path):
    """With 2 replicas and reserved=1, a held follow-up leaves one free for TTFA."""
    piper_pool.clear_pool_for_tests()
    model = tmp_path / "v.onnx"
    model.write_bytes(b"onnx")
    (tmp_path / "v.onnx.json").write_text(
        '{"num_speakers": 1, "speaker_id_map": {}}',
        encoding="utf-8",
    )

    class FakeVoice:
        def __init__(self):
            self.config = MagicMock(sample_rate=22050)
            self.session = object()

        def synthesize(self, text, syn_config=None):
            chunk = MagicMock(
                sample_rate=22050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=b"\x00\x00" * 10,
            )
            yield chunk

    monkeypatch.setattr(piper_pool, "_create_piper_voice", lambda p: FakeVoice())
    monkeypatch.setenv("PIPER_POOL_SLOTS", "2")
    monkeypatch.setenv("PIPER_FIRST_CHUNK_RESERVED", "1")

    piper_pool.warm_voice_models([model], wait_for_ready=True, timeout_seconds=5.0)
    assert sum(piper_pool.model_replica_counts_for_tests().values()) >= 2

    # Check out one instance as a follow-up (simulates in-flight remaining clip).
    follow = piper_pool._acquire_instance(model, first_chunk=False)
    assert follow is not None

    first_got = threading.Event()
    first_instance: list = []

    def first_waiter():
        inst = piper_pool._acquire_instance(model, first_chunk=True)
        first_instance.append(inst)
        first_got.set()

    t = threading.Thread(target=first_waiter)
    t.start()
    assert first_got.wait(timeout=2.0), "first-chunk acquire should get the reserved free replica"
    assert first_instance[0] is not None

    # Second follow-up should not steal while first-chunk is still holding / waiting path
    # Release first so pool can drain; then ensure follow-up can proceed when no waiters.
    piper_pool._release_instance(first_instance[0])
    piper_pool._release_instance(follow)
    t.join(timeout=1.0)


def test_followup_blocked_when_only_reserved_free(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model = tmp_path / "v.onnx"
    model.write_bytes(b"onnx")
    (tmp_path / "v.onnx.json").write_text(
        '{"num_speakers": 1, "speaker_id_map": {}}',
        encoding="utf-8",
    )

    class FakeVoice:
        def __init__(self):
            self.config = MagicMock(sample_rate=22050)
            self.session = object()

    monkeypatch.setattr(piper_pool, "_create_piper_voice", lambda p: FakeVoice())
    monkeypatch.setenv("PIPER_POOL_SLOTS", "2")
    monkeypatch.setenv("PIPER_FIRST_CHUNK_RESERVED", "1")
    piper_pool.warm_voice_models([model], wait_for_ready=True, timeout_seconds=5.0)

    held = piper_pool._acquire_instance(model, first_chunk=True)
    entry = next(iter(piper_pool._MODELS.values()))
    assert entry.free.qsize() == 1
    with piper_pool._METRICS_LOCK:
        piper_pool._waiting_first_chunk = 1
    assert piper_pool._followup_may_take(entry) is False
    with piper_pool._METRICS_LOCK:
        piper_pool._waiting_first_chunk = 0
    assert piper_pool._followup_may_take(entry) is True

    piper_pool._release_instance(held)


def test_synthesize_wav_file_passes_first_chunk_flag(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model = tmp_path / "v.onnx"
    model.write_bytes(b"onnx")
    (tmp_path / "v.onnx.json").write_text(
        '{"num_speakers": 1, "speaker_id_map": {}}',
        encoding="utf-8",
    )
    seen: list[bool] = []

    class FakeVoice:
        def __init__(self):
            self.config = MagicMock(sample_rate=22050)

        def synthesize(self, text, syn_config=None):
            chunk = MagicMock(
                sample_rate=22050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=b"\x00\x00" * 10,
            )
            yield chunk

    real_acquire = piper_pool._acquire_instance

    def spy_acquire(path, *, first_chunk=True):
        seen.append(first_chunk)
        return real_acquire(path, first_chunk=first_chunk)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", lambda p: FakeVoice())
    monkeypatch.setattr(piper_pool, "_acquire_instance", spy_acquire)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")
    monkeypatch.setenv("PIPER_FIRST_CHUNK_RESERVED", "0")
    piper_pool.warm_voice_models([model], wait_for_ready=True, timeout_seconds=5.0)

    out = tmp_path / "o.wav"
    piper_pool.synthesize_wav_file(
        model,
        out,
        "hi",
        length_scale=1.0,
        noise_scale=0.667,
        noise_w=0.8,
        sentence_silence=0.0,
        first_chunk=False,
    )
    assert seen == [False]
    assert out.is_file()
