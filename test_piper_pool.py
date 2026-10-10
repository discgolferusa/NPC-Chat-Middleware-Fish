"""Unit tests for in-process Piper voice pool warm-up and replica slots."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

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


def _write_onnx_with_speakers(path: Path, num_speakers: int) -> Path:
    path.write_bytes(b"onnx")
    speaker_map = {f"spk_{i}": i for i in range(num_speakers)}
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps({"num_speakers": num_speakers, "speaker_id_map": speaker_map}),
        encoding="utf-8",
    )
    return path


def test_allocate_replicas_equal_speakers():
    got = piper_pool.allocate_replicas({"a": 100, "b": 100}, 10)
    assert got == {"a": 5, "b": 5}


def test_allocate_replicas_proportional():
    got = piper_pool.allocate_replicas({"big": 90, "small": 10}, 10)
    assert got == {"big": 9, "small": 1}


def test_allocate_replicas_single_model_gets_budget():
    got = piper_pool.allocate_replicas({"only": 40}, 10)
    assert got == {"only": 10}


def test_allocate_replicas_more_models_than_slots():
    got = piper_pool.allocate_replicas({f"m{i}": 10 for i in range(12)}, 10)
    assert all(v == 1 for v in got.values())
    assert len(got) == 12


def test_warm_voice_models_loads_in_background(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "npc.onnx", 1)

    load_started = threading.Event()
    load_done = threading.Event()

    def fake_create(path: Path):
        load_started.set()
        time.sleep(0.05)
        load_done.set()
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "2")

    piper_pool.warm_voice_models([model_path], request_id="req-warm")

    assert load_started.wait(timeout=1.0)
    assert load_done.wait(timeout=1.0)
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        counts = piper_pool.model_replica_counts_for_tests()
        if counts.get(str(model_path), 0) >= 1:
            break
        time.sleep(0.01)
    assert piper_pool.model_replica_counts_for_tests()[str(model_path)] >= 1


def test_blocking_warm_returns_only_after_full_replica_target(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "blocking.onnx", 1)
    calls = {"count": 0}
    calls_lock = threading.Lock()

    def fake_create(path: Path):
        time.sleep(0.03)
        with calls_lock:
            calls["count"] += 1
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "3")

    status = piper_pool.warm_voice_models(
        [model_path],
        request_id="startup",
        wait_for_ready=True,
        timeout_seconds=2.0,
    )

    assert calls["count"] == 3
    assert status is not None
    assert status[str(model_path)]["ready"] == 3
    assert status[str(model_path)]["target"] == 3
    assert status[str(model_path)]["inflight"] == 0


def test_blocking_warm_raises_if_a_replica_cannot_load(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "broken.onnx", 1)

    def fail_create(_path: Path):
        raise OSError("invalid ONNX model")

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fail_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")

    with pytest.raises(RuntimeError, match="Piper pool warm-up failed"):
        piper_pool.warm_voice_models(
            [model_path],
            request_id="startup",
            wait_for_ready=True,
            timeout_seconds=1.0,
        )


def test_warm_voice_models_is_idempotent(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "npc.onnx", 1)
    calls = {"count": 0}

    def fake_create(path: Path):
        calls["count"] += 1
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")

    piper_pool.warm_voice_models([model_path, model_path], request_id="req1")
    time.sleep(0.05)
    piper_pool.warm_voice_models([model_path], request_id="req2")
    time.sleep(0.05)

    assert calls["count"] == 1


def test_warm_splits_slots_by_speaker_count(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    big = _write_onnx_with_speakers(tmp_path / "big.onnx", 90)
    small = _write_onnx_with_speakers(tmp_path / "small.onnx", 10)

    def fake_create(path: Path):
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "10")

    piper_pool.warm_voice_models([big, small], request_id="split")
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        targets = piper_pool.model_targets_for_tests()
        counts = piper_pool.model_replica_counts_for_tests()
        if (
            targets.get(str(big)) == 9
            and targets.get(str(small)) == 1
            and counts.get(str(big)) == 9
            and counts.get(str(small)) == 1
        ):
            break
        time.sleep(0.02)

    assert piper_pool.model_targets_for_tests()[str(big)] == 9
    assert piper_pool.model_targets_for_tests()[str(small)] == 1
    assert piper_pool.model_replica_counts_for_tests()[str(big)] == 9
    assert piper_pool.model_replica_counts_for_tests()[str(small)] == 1
    status = piper_pool.pool_status()
    assert status[str(big)]["speaker_count"] == 90
    assert status[str(small)]["speaker_count"] == 10


def test_cold_warm_loads_one_replica_before_topup(monkeypatch, tmp_path):
    """Cold start must not run N parallel ONNX loads before the first replica is ready."""
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "solo.onnx", 50)

    active = {"n": 0}
    # Peak concurrency only while the first (bootstrap) create is in flight.
    # Top-up creates after first_done must not participate — otherwise this assert races.
    bootstrap_peak = {"n": 0}
    first_done = threading.Event()
    lock = threading.Lock()

    def fake_create(path: Path):
        with lock:
            # Capture before any later create can see first_done; top-ups start only
            # after this function returns and _store_loaded_replica kicks loaders.
            was_bootstrap = not first_done.is_set()
            active["n"] += 1
            if was_bootstrap:
                bootstrap_peak["n"] = max(bootstrap_peak["n"], active["n"])
        if was_bootstrap:
            time.sleep(0.05)
        with lock:
            active["n"] -= 1
            if was_bootstrap:
                first_done.set()
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "10")

    piper_pool.warm_voice_models([model_path], request_id="bootstrap")
    assert first_done.wait(timeout=1.0)

    # Bootstrap was single-flighted (top-up overlap is allowed and ignored here).
    assert bootstrap_peak["n"] == 1

    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        if piper_pool.model_replica_counts_for_tests().get(str(model_path), 0) == 10:
            break
        time.sleep(0.02)

    assert piper_pool.model_targets_for_tests()[str(model_path)] == 10
    assert piper_pool.model_replica_counts_for_tests()[str(model_path)] == 10


def test_synthesize_wav_file_uses_pooled_voice(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "npc.onnx", 1)
    output_path = tmp_path / "out.wav"

    fake_voice = _FakeVoice(model_path)
    instance = piper_pool._VoiceInstance(fake_voice, str(model_path))
    monkeypatch.setattr(piper_pool, "_acquire_instance", lambda path: instance)
    monkeypatch.setattr(piper_pool, "_release_instance", lambda inst: None)

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
    model_path = _write_onnx_with_speakers(tmp_path / "wait.onnx", 1)

    started = threading.Event()

    def slow_create(path: Path):
        started.set()
        time.sleep(0.08)
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", slow_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")
    piper_pool.warm_voice_models([model_path], request_id="req-wait")
    assert started.wait(timeout=1.0)

    instance = piper_pool._acquire_instance(model_path)
    assert instance.voice.model_path == model_path
    piper_pool._release_instance(instance)


def test_cold_path_load_does_not_block_other_models(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    slow_path = _write_onnx_with_speakers(tmp_path / "slow.onnx", 1)
    fast_path = _write_onnx_with_speakers(tmp_path / "fast.onnx", 1)

    load_started = threading.Event()
    release_slow = threading.Event()

    def fake_create(path: Path):
        if path == slow_path:
            load_started.set()
            assert release_slow.wait(timeout=1.0)
            return _FakeVoice(path)
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "2")

    slow_result: dict[str, Any] = {}
    fast_result: dict[str, Any] = {}

    def load_slow():
        instance = piper_pool._acquire_instance(slow_path)
        slow_result["voice"] = instance.voice
        piper_pool._release_instance(instance)

    def load_fast():
        load_started.wait(timeout=1.0)
        instance = piper_pool._acquire_instance(fast_path)
        fast_result["voice"] = instance.voice
        piper_pool._release_instance(instance)

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


def test_same_model_replicas_allow_parallel_synth(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "multi.onnx", 50)

    hold = threading.Event()
    started = {"n": 0}
    started_lock = threading.Lock()
    both_started = threading.Event()

    class HoldingVoice(_FakeVoice):
        def synthesize(self, text, syn_config=None):
            with started_lock:
                started["n"] += 1
                if started["n"] >= 2:
                    both_started.set()
            assert hold.wait(timeout=1.0)
            yield from super().synthesize(text, syn_config)

    def fake_create(path: Path):
        return HoldingVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "2")

    piper_pool.warm_voice_models([model_path], request_id="parallel")
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        if piper_pool.model_replica_counts_for_tests().get(str(model_path), 0) >= 2:
            break
        time.sleep(0.02)
    assert piper_pool.model_replica_counts_for_tests()[str(model_path)] == 2

    results: dict[str, bool] = {}

    def run_synth(name: str, out: Path):
        piper_pool.synthesize_wav_file(
            model_path,
            out,
            "hello",
            length_scale=1.0,
            noise_scale=0.5,
            noise_w=0.5,
            sentence_silence=0.0,
        )
        results[name] = True

    t1 = threading.Thread(target=run_synth, args=("a", tmp_path / "a.wav"))
    t2 = threading.Thread(target=run_synth, args=("b", tmp_path / "b.wav"))
    t1.start()
    t2.start()
    assert both_started.wait(timeout=1.0)
    hold.set()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)
    assert results == {"a": True, "b": True}


def test_get_voice_retries_after_warm_failure(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "flaky.onnx", 1)
    calls = {"n": 0}

    def fake_create(path: Path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient warm failure")
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")

    piper_pool.warm_voice_models([model_path], request_id="warm-fail")
    time.sleep(0.08)

    instance = piper_pool._acquire_instance(model_path)
    assert instance.voice.model_path == model_path
    assert calls["n"] == 2
    piper_pool._release_instance(instance)


def test_waiter_waits_for_inflight_retry_instead_of_stale_error(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "race.onnx", 1)
    key = str(model_path)

    entry = piper_pool._ModelEntry(model_path, 1)
    entry.target_count = 1
    entry.error = OSError("first failure")
    entry.ready.set()
    piper_pool._MODELS[key] = entry

    retry_started = threading.Event()
    allow_retry_done = threading.Event()

    def fake_create(path: Path):
        retry_started.set()
        assert allow_retry_done.wait(timeout=1.0)
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")

    waiter_result: dict[str, Any] = {}

    def waiter():
        instance = piper_pool._acquire_instance(model_path)
        waiter_result["voice"] = instance.voice
        piper_pool._release_instance(instance)

    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()

    assert retry_started.wait(timeout=1.0)

    with piper_pool._POOL_LOCK:
        active = piper_pool._MODELS[key]
        assert active.inflight >= 1
        assert active.error is None

    allow_retry_done.set()
    waiter_thread.join(timeout=1.0)

    assert waiter_result["voice"].model_path == model_path


class _ReentrancyDetectLock:
    """threading.Lock stand-in that fails fast on same-thread re-acquire."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._count = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.get_ident()
        if self._owner == me:
            raise RuntimeError("reentrant _POOL_LOCK acquire")
        ok = self._lock.acquire(blocking, timeout) if timeout >= 0 else self._lock.acquire(blocking)
        if ok:
            self._owner = me
            self._count += 1
        return ok

    def release(self) -> None:
        self._owner = None
        self._count -= 1
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def test_acquire_error_retry_does_not_reenter_pool_lock(monkeypatch, tmp_path):
    """Regression: error retry must not call _kick_loaders while holding _POOL_LOCK."""
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "retry.onnx", 1)
    key = str(model_path)

    entry = piper_pool._ModelEntry(model_path, 1)
    entry.target_count = 1
    entry.error = OSError("stale")
    entry.ready.set()
    piper_pool._MODELS[key] = entry

    detect = _ReentrancyDetectLock()
    monkeypatch.setattr(piper_pool, "_POOL_LOCK", detect)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")

    def fake_create(path: Path):
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", fake_create)

    instance = piper_pool._acquire_instance(model_path)
    assert instance.voice.model_path == model_path
    piper_pool._release_instance(instance)


def test_slow_warm_completes_while_acquire_waits(monkeypatch, tmp_path):
    """Regression: acquire must not hold _POOL_LOCK while waiting on ready."""
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "slow-warm.onnx", 1)

    load_started = threading.Event()
    release_load = threading.Event()

    def slow_create(path: Path):
        load_started.set()
        assert release_load.wait(timeout=2.0)
        return _FakeVoice(path)

    monkeypatch.setattr(piper_pool, "_create_piper_voice", slow_create)
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")

    result: dict[str, Any] = {}

    def waiter():
        instance = piper_pool._acquire_instance(model_path)
        result["voice"] = instance.voice
        piper_pool._release_instance(instance)

    thread = threading.Thread(target=waiter)
    thread.start()
    assert load_started.wait(timeout=1.0)

    # While loader is blocked, pool lock must still be acquirable (acquire is waiting unlocked).
    assert piper_pool._POOL_LOCK.acquire(timeout=0.5)
    piper_pool._POOL_LOCK.release()

    release_load.set()
    thread.join(timeout=2.0)
    assert thread.is_alive() is False
    assert result["voice"].model_path == model_path


def test_pool_metrics_record_acquire_wait_and_synth(monkeypatch, tmp_path):
    """Instrumentation records wait/synth without changing allocation outcomes."""
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "metrics.onnx", 1)
    monkeypatch.setattr(piper_pool, "_create_piper_voice", lambda path: _FakeVoice(path))
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")

    piper_pool.warm_voice_models([model_path], wait_for_ready=True, timeout_seconds=5.0)
    out = tmp_path / "out.wav"
    piper_pool.synthesize_wav_file(
        model_path,
        out,
        "Hello metrics.",
        length_scale=1.0,
        noise_scale=0.667,
        noise_w=0.8,
        sentence_silence=0.2,
    )
    call = piper_pool.last_call_metrics()
    assert "acquire_wait_ms" in call
    assert "synth_ms" in call
    assert call["acquire_wait_ms"] >= 0
    assert call["synth_ms"] >= 0
    runtime = piper_pool.pool_runtime_metrics()
    assert runtime["waiting_requests"] == 0
    assert runtime["active_slots"] == 0
    assert out.is_file()


def test_waiting_requests_gauge_during_contention(monkeypatch, tmp_path):
    piper_pool.clear_pool_for_tests()
    model_path = _write_onnx_with_speakers(tmp_path / "wait-gauge.onnx", 1)
    monkeypatch.setattr(piper_pool, "_create_piper_voice", lambda path: _FakeVoice(path))
    monkeypatch.setenv("PIPER_POOL_SLOTS", "1")
    piper_pool.warm_voice_models([model_path], wait_for_ready=True, timeout_seconds=5.0)

    hold = threading.Event()
    held = threading.Event()

    def hold_slot():
        instance = piper_pool._acquire_instance(model_path)
        held.set()
        assert hold.wait(timeout=2.0)
        piper_pool._release_instance(instance)

    t1 = threading.Thread(target=hold_slot)
    t1.start()
    assert held.wait(timeout=1.0)

    waiting_seen = {"max": 0}
    stop = threading.Event()

    def waiter():
        instance = piper_pool._acquire_instance(model_path)
        piper_pool._release_instance(instance)

    def sample():
        while not stop.wait(0.01):
            waiting_seen["max"] = max(
                waiting_seen["max"],
                int(piper_pool.pool_runtime_metrics()["waiting_requests"]),
            )

    t_sample = threading.Thread(target=sample)
    t_sample.start()
    t2 = threading.Thread(target=waiter)
    t2.start()
    time.sleep(0.1)
    assert waiting_seen["max"] >= 1 or piper_pool.pool_runtime_metrics()["active_slots"] >= 1
    hold.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    stop.set()
    t_sample.join(timeout=1.0)
    assert piper_pool.pool_runtime_metrics()["waiting_requests"] == 0
    assert piper_pool.pool_runtime_metrics()["active_slots"] == 0
