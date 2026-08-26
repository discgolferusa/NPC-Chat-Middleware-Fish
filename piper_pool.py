"""In-process Piper voice pool with background warm-up and multi-replica slots."""

from __future__ import annotations

import os
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Any

_POOL_LOCK = threading.Lock()
_DEFAULT_SLOT_BUDGET = 10


class _VoiceInstance:
    __slots__ = ("voice", "model_key")

    def __init__(self, voice: Any, model_key: str) -> None:
        self.voice = voice
        self.model_key = model_key


class _ModelEntry:
    __slots__ = (
        "model_path",
        "weight",
        "target_count",
        "instances",
        "free",
        "inflight",
        "ready",
        "error",
    )

    def __init__(self, model_path: Path, weight: int) -> None:
        self.model_path = model_path
        self.weight = max(1, int(weight))
        self.target_count = 1
        self.instances: list[_VoiceInstance] = []
        self.free: queue.Queue[_VoiceInstance] = queue.Queue()
        self.inflight = 0
        self.ready = threading.Event()
        self.error: BaseException | None = None


_MODELS: dict[str, _ModelEntry] = {}


def _voice_key(model_path: Path) -> str:
    return str(model_path)


def pool_slot_budget() -> int:
    raw = str(os.environ.get("PIPER_POOL_SLOTS", "")).strip()
    if not raw:
        return _DEFAULT_SLOT_BUDGET
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_SLOT_BUDGET


def allocate_replicas(weights: dict[str, int], total_slots: int) -> dict[str, int]:
    """
    Largest-remainder share of ``total_slots`` by speaker weight.

    Each registered model gets at least 1 when ``len(weights) <= total_slots``.
    If more models than slots, each still gets 1 (soft over-budget) so none starve.
    """
    if not weights:
        return {}
    keys = list(weights.keys())
    n = len(keys)
    cleaned = {k: max(1, int(weights[k] or 1)) for k in keys}
    budget = max(1, int(total_slots))

    if n >= budget:
        return {k: 1 for k in keys}

    total_w = sum(cleaned.values())
    exact = {k: budget * cleaned[k] / total_w for k in keys}
    result = {k: int(exact[k]) for k in keys}

    for k in keys:
        if result[k] > 0:
            continue
        donor = max(keys, key=lambda x: result[x])
        if result[donor] > 1:
            result[donor] -= 1
            result[k] = 1
        else:
            result[k] = 1

    used = sum(result.values())
    rem = budget - used
    if rem > 0:
        order = sorted(
            keys,
            key=lambda k: (exact[k] - int(exact[k]), cleaned[k]),
            reverse=True,
        )
        for k in order:
            if rem <= 0:
                break
            result[k] += 1
            rem -= 1
    elif rem < 0:
        order = sorted(keys, key=lambda k: result[k], reverse=True)
        guard = 0
        while rem < 0 and guard < n * budget:
            k = order[guard % n]
            if result[k] > 1:
                result[k] -= 1
                rem += 1
            guard += 1

    return result


def _speaker_weight(model_path: Path) -> int:
    from tts_processing import load_piper_model_meta

    meta = load_piper_model_meta(model_path)
    return max(1, int(meta.get("num_speakers") or 1))


def _create_piper_voice(model_path: Path) -> Any:
    from piper import PiperVoice

    if not model_path.is_file():
        raise FileNotFoundError(f"Piper voice model not found: {model_path}")

    use_cuda = str(os.environ.get("PIPER_USE_CUDA", "")).strip().lower() in ("1", "true", "yes")
    return PiperVoice.load(model_path, use_cuda=use_cuda)


def _trim_idle_surplus_locked(entry: _ModelEntry) -> None:
    """Drop idle free replicas when target shrank after rebalance."""
    while len(entry.instances) > entry.target_count:
        try:
            instance = entry.free.get_nowait()
        except queue.Empty:
            break
        if instance in entry.instances:
            entry.instances.remove(instance)


def _recompute_targets_locked() -> None:
    weights = {key: entry.weight for key, entry in _MODELS.items()}
    targets = allocate_replicas(weights, pool_slot_budget())
    for key, entry in _MODELS.items():
        entry.target_count = targets.get(key, 1)
        _trim_idle_surplus_locked(entry)


def _capacity_count(entry: _ModelEntry) -> int:
    return len(entry.instances) + entry.inflight


def _register_model_locked(model_path: Path) -> _ModelEntry:
    key = _voice_key(model_path)
    entry = _MODELS.get(key)
    if entry is not None:
        return entry
    entry = _ModelEntry(model_path, _speaker_weight(model_path))
    _MODELS[key] = entry
    _recompute_targets_locked()
    return entry


def _start_loaders_locked(entry: _ModelEntry, request_id: str | None) -> list[threading.Thread]:
    """
    Start replica load threads up to the model target.

    Cold path: only one loader until the first instance exists, so first-request
    TTS is not slowed by N parallel ONNX loads contending for disk/RAM/CPU.
    After the first replica is ready, callers top up to ``target_count``.
    """
    threads: list[threading.Thread] = []
    if not entry.instances:
        if entry.inflight > 0:
            return threads
        desired = 1
    else:
        desired = entry.target_count

    while _capacity_count(entry) < desired:
        entry.inflight += 1
        entry.error = None
        if not entry.instances:
            entry.ready.clear()
        thread = threading.Thread(
            target=_load_voice_replica,
            args=(entry.model_path, request_id),
            name=f"piper-warm-{entry.model_path.name}",
            daemon=True,
        )
        threads.append(thread)
    return threads


def _store_loaded_replica(key: str, voice: Any) -> list[threading.Thread]:
    """Store a loaded replica and return any top-up loader threads to start."""
    with _POOL_LOCK:
        entry = _MODELS.get(key)
        if entry is None:
            return []
        entry.inflight = max(0, entry.inflight - 1)
        if len(entry.instances) >= entry.target_count:
            # Target shrank while this replica was loading; discard it.
            entry.error = None
            if entry.instances:
                entry.ready.set()
            return []
        instance = _VoiceInstance(voice, key)
        entry.instances.append(instance)
        entry.error = None
        entry.ready.set()
        entry.free.put(instance)
        # First replica is ready — fill remaining slots in the background.
        return _start_loaders_locked(entry, None)


def _store_load_error(key: str, error: BaseException) -> None:
    with _POOL_LOCK:
        entry = _MODELS.get(key)
        if entry is None:
            return
        entry.inflight = max(0, entry.inflight - 1)
        if not entry.instances and entry.inflight == 0:
            entry.error = error
            entry.ready.set()


def _load_voice_replica(model_path: Path, request_id: str | None) -> None:
    key = _voice_key(model_path)
    try:
        voice = _create_piper_voice(model_path)
        topup_threads = _store_loaded_replica(key, voice)
        for thread in topup_threads:
            thread.start()
        with _POOL_LOCK:
            entry = _MODELS.get(key)
            live = len(entry.instances) if entry is not None else 0
            target = entry.target_count if entry is not None else 0
        print(
            f"[PIPER-POOL] warmed model={model_path.name} "
            f"replicas={live}/{target} request_id={request_id or '-'}",
            flush=True,
        )
    except Exception as ex:
        _store_load_error(key, ex)
        print(
            f"[PIPER-POOL] warm_failed model={model_path.name} "
            f"request_id={request_id or '-'} error={ex!r}",
            flush=True,
        )


def warm_voice_models(model_paths: list[Path], *, request_id: str | None = None) -> None:
    """Register models, rebalance slot targets, and warm replicas in the background."""
    unique: list[Path] = []
    seen: set[str] = set()
    for raw in model_paths:
        if raw is None:
            continue
        path = Path(raw)
        key = _voice_key(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    threads: list[threading.Thread] = []
    with _POOL_LOCK:
        for model_path in unique:
            _register_model_locked(model_path)
        _recompute_targets_locked()
        for entry in _MODELS.values():
            threads.extend(_start_loaders_locked(entry, request_id))

    for thread in threads:
        thread.start()


def _kick_loaders(model_path: Path, *, clear_error: bool = False) -> _ModelEntry:
    with _POOL_LOCK:
        entry = _register_model_locked(model_path)
        _recompute_targets_locked()
        if clear_error and entry.error is not None and not entry.instances:
            entry.error = None
            entry.ready.clear()
        threads = _start_loaders_locked(entry, None)
    for thread in threads:
        thread.start()
    return entry


def _snapshot_acquire_state(key: str) -> tuple[bool, BaseException | None, int, threading.Event] | None:
    """Read acquire-relevant fields under the pool lock; never block/wait here."""
    with _POOL_LOCK:
        current = _MODELS.get(key)
        if current is None:
            return None
        return (
            bool(current.instances),
            current.error,
            current.inflight,
            current.ready,
        )


def _acquire_instance(model_path: Path) -> _VoiceInstance:
    key = _voice_key(model_path)
    deadline = time.perf_counter() + 120.0
    retried_error = False

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for Piper voice load: {model_path}")

        entry = _kick_loaders(model_path)

        snap = _snapshot_acquire_state(key)
        if snap is None:
            continue
        has_instances, err, inflight, ready_event = snap
        if not has_instances and err is not None and inflight == 0:
            if retried_error:
                raise err
            retried_error = True
            # Must not hold _POOL_LOCK here — _kick_loaders takes it (non-reentrant).
            entry = _kick_loaders(model_path, clear_error=True)

        try:
            return entry.free.get(timeout=min(0.5, remaining))
        except queue.Empty:
            snap = _snapshot_acquire_state(key)
            if snap is None:
                continue
            has_instances, err, inflight, ready_event = snap
            if not has_instances and err is not None and inflight == 0:
                if retried_error:
                    raise err
                retried_error = True
                _kick_loaders(model_path, clear_error=True)
                continue
            if not has_instances:
                # Wait outside _POOL_LOCK so loaders can store replicas / set ready.
                ready_event.wait(timeout=min(0.5, remaining))
            continue


def _release_instance(instance: _VoiceInstance) -> None:
    with _POOL_LOCK:
        entry = _MODELS.get(instance.model_key)
        if entry is None:
            return
        if len(entry.instances) > entry.target_count and instance in entry.instances:
            entry.instances.remove(instance)
            return
    entry.free.put(instance)


def synthesize_wav_file(
    model_path: Path,
    output_path: Path,
    text: str,
    *,
    length_scale: float,
    noise_scale: float,
    noise_w: float,
    sentence_silence: float,
    speaker_id: int | None = None,
) -> None:
    """Synthesize text to a WAV file using a pooled in-process Piper voice replica."""
    from piper import SynthesisConfig

    instance = _acquire_instance(model_path)
    voice = instance.voice
    try:
        syn_config = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
            normalize_audio=True,
        )
        silence_bytes = bytes(int(voice.config.sample_rate * sentence_silence * 2))

        with wave.open(str(output_path), "wb") as wav_file:
            wav_params_set = False
            for chunk_index, audio_chunk in enumerate(voice.synthesize(text, syn_config)):
                if not wav_params_set:
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)
                    wav_params_set = True

                if chunk_index > 0:
                    wav_file.writeframes(silence_bytes)

                wav_file.writeframes(audio_chunk.audio_int16_bytes)
    finally:
        _release_instance(instance)


def clear_pool_for_tests() -> None:
    """Reset module state (tests only)."""
    with _POOL_LOCK:
        _MODELS.clear()


def model_replica_counts_for_tests() -> dict[str, int]:
    """Return live replica counts keyed by model path (tests only)."""
    with _POOL_LOCK:
        return {key: len(entry.instances) for key, entry in _MODELS.items()}


def model_targets_for_tests() -> dict[str, int]:
    """Return target replica counts keyed by model path (tests only)."""
    with _POOL_LOCK:
        return {key: entry.target_count for key, entry in _MODELS.items()}
