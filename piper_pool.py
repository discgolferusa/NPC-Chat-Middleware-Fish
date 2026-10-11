"""In-process Piper voice pool with background warm-up and multi-replica slots."""

from __future__ import annotations

import os
import queue
import threading
import time
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Any

_POOL_LOCK = threading.Lock()
_DEFAULT_SLOT_BUDGET = 10
_DEFAULT_FIRST_CHUNK_RESERVED = 2
_DEFAULT_PHONEME_CACHE_SIZE = 256

# Instrumentation gauges (do not affect scheduling). Protected by _METRICS_LOCK
# except thread-local last-call fields.
_METRICS_LOCK = threading.Lock()
_waiting_acquires = 0
_waiting_first_chunk = 0
_checked_out = 0
_tls = threading.local()

# Process-wide phoneme LRU (espeak is process-serialized; cache avoids repeat work).
_PHONEME_CACHE_LOCK = threading.Lock()
_PHONEME_CACHE: OrderedDict[tuple[str, str], list] = OrderedDict()
_phoneme_cache_hits = 0
_phoneme_cache_misses = 0


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


def first_chunk_reserved_slots() -> int:
    """
    Replicas kept preferentially for first-chunk (TTFA) acquires.

    ``PIPER_FIRST_CHUNK_RESERVED`` unset → default 2.
    ``0`` disables reservation (legacy equal scheduling).
    Capped so at least one replica remains usable for follow-ups when budget > 1.
    """
    raw = str(os.environ.get("PIPER_FIRST_CHUNK_RESERVED", "")).strip()
    if not raw:
        reserved = _DEFAULT_FIRST_CHUNK_RESERVED
    else:
        try:
            reserved = max(0, int(raw))
        except ValueError:
            reserved = _DEFAULT_FIRST_CHUNK_RESERVED
    budget = pool_slot_budget()
    if budget <= 1:
        return 0
    return min(reserved, budget - 1)


def parse_model_slot_budgets(raw: str | None = None) -> dict[str, int]:
    """
    Parse ``PIPER_MODEL_SLOT_BUDGETS`` like ``en_US-libritts-high:8,lessac:2``.

    Keys are matched case-insensitively against model file stems (and full stem
    prefixes). Values are replica counts (>= 1).
    """
    text = str(raw if raw is not None else os.environ.get("PIPER_MODEL_SLOT_BUDGETS", "")).strip()
    if not text:
        return {}
    out: dict[str, int] = {}
    for part in text.split(","):
        piece = part.strip()
        if not piece or ":" not in piece:
            continue
        name, _, count_s = piece.partition(":")
        key = name.strip().lower()
        if not key:
            continue
        try:
            count = max(1, int(count_s.strip()))
        except ValueError:
            continue
        out[key] = count
    return out


def _match_budget_key(model_path: Path, budgets: dict[str, int]) -> int | None:
    if not budgets:
        return None
    stem = model_path.stem.lower()
    if stem in budgets:
        return budgets[stem]
    # Allow partial keys (e.g. "libritts" → en_US-libritts-high).
    for key, count in budgets.items():
        if key in stem:
            return count
    return None


def apply_explicit_slot_budgets(
    model_keys: list[str],
    total_slots: int,
    budgets: dict[str, int] | None = None,
) -> dict[str, int] | None:
    """
    Build per-model targets from explicit budgets.

    Returns None when no budgets apply (caller should use speaker-weight allocation).
    Unlisted models get at least 1 if slots remain; excess budget is clamped to
    ``total_slots``.
    """
    parsed = budgets if budgets is not None else parse_model_slot_budgets()
    if not parsed or not model_keys:
        return None

    budget = max(1, int(total_slots))
    assigned: dict[str, int] = {}
    matched = 0
    for key in model_keys:
        hit = _match_budget_key(Path(key), parsed)
        if hit is not None:
            assigned[key] = hit
            matched += 1
        else:
            assigned[key] = 0

    if matched == 0:
        return None

    # Ensure every model has at least 1 when possible.
    for key in model_keys:
        if assigned[key] < 1:
            assigned[key] = 1

    used = sum(assigned.values())
    if used > budget:
        # Scale down largest first until within budget, never below 1.
        while used > budget:
            donor = max(model_keys, key=lambda k: assigned[k])
            if assigned[donor] <= 1:
                break
            assigned[donor] -= 1
            used -= 1
    elif used < budget:
        # Give remainder to models that appeared in the explicit map (prefer those).
        prefer = [k for k in model_keys if _match_budget_key(Path(k), parsed) is not None]
        order = prefer + [k for k in model_keys if k not in prefer]
        rem = budget - used
        i = 0
        while rem > 0 and order:
            assigned[order[i % len(order)]] += 1
            rem -= 1
            i += 1

    return assigned


def allocate_replicas(weights: dict[str, int], total_slots: int) -> dict[str, int]:
    """
    Largest-remainder share of ``total_slots`` by speaker weight.

    When ``PIPER_MODEL_SLOT_BUDGETS`` matches registered models, those explicit
    targets win (see ``apply_explicit_slot_budgets``).

    Each registered model gets at least 1 when ``len(weights) <= total_slots``.
    If more models than slots, each still gets 1 (soft over-budget) so none starve.
    """
    if not weights:
        return {}
    keys = list(weights.keys())
    explicit = apply_explicit_slot_budgets(keys, total_slots)
    if explicit is not None:
        return explicit
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


def phoneme_cache_max_size() -> int:
    """``PIPER_PHONEME_CACHE_SIZE`` (default 256). ``0`` disables caching."""
    raw = str(os.environ.get("PIPER_PHONEME_CACHE_SIZE", "")).strip()
    if not raw:
        return _DEFAULT_PHONEME_CACHE_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_PHONEME_CACHE_SIZE


def phoneme_cache_stats() -> dict[str, int]:
    with _PHONEME_CACHE_LOCK:
        return {
            "size": len(_PHONEME_CACHE),
            "max_size": phoneme_cache_max_size(),
            "hits": int(_phoneme_cache_hits),
            "misses": int(_phoneme_cache_misses),
        }


def clear_phoneme_cache_for_tests() -> None:
    global _phoneme_cache_hits, _phoneme_cache_misses
    with _PHONEME_CACHE_LOCK:
        _PHONEME_CACHE.clear()
        _phoneme_cache_hits = 0
        _phoneme_cache_misses = 0


def _wrap_voice_phonemize_cache(voice: Any, model_key: str) -> Any:
    """Attach an LRU cache around ``voice.phonemize`` when cache size > 0."""
    max_size = phoneme_cache_max_size()
    if max_size <= 0:
        return voice
    original = voice.phonemize

    def cached_phonemize(text: str):
        global _phoneme_cache_hits, _phoneme_cache_misses
        key = (model_key, str(text))
        with _PHONEME_CACHE_LOCK:
            hit = _PHONEME_CACHE.get(key)
            if hit is not None:
                _PHONEME_CACHE.move_to_end(key)
                _phoneme_cache_hits += 1
                # Return a shallow copy of the outer list to avoid caller mutation.
                return [list(part) for part in hit]
            _phoneme_cache_misses += 1
        phonemes = original(text)
        with _PHONEME_CACHE_LOCK:
            _PHONEME_CACHE[key] = [list(part) for part in phonemes]
            _PHONEME_CACHE.move_to_end(key)
            while len(_PHONEME_CACHE) > max_size:
                _PHONEME_CACHE.popitem(last=False)
        return phonemes

    voice.phonemize = cached_phonemize  # type: ignore[method-assign]
    return voice


def onnx_intra_op_threads() -> int:
    """
    Optional ORT intra-op thread cap for each Piper InferenceSession.

    ``PIPER_ONNX_INTRA_OP_THREADS`` unset or ``0`` → Piper/ORT automatic behavior
    (same as stock ``PiperVoice.load`` / default ``SessionOptions``).
    Values ``>= 1`` pin ``SessionOptions.intra_op_num_threads`` at session create.
    """
    raw = str(os.environ.get("PIPER_ONNX_INTRA_OP_THREADS", "")).strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _create_piper_voice(model_path: Path) -> Any:
    """
    Load a Piper voice for the pool.

    ``PiperVoice.load()`` does not expose ONNX ``SessionOptions``. When
    ``PIPER_ONNX_INTRA_OP_THREADS`` is 0/unset we call ``load()`` unchanged.
    When set to a positive integer we construct the same ``PiperVoice`` shape
    as upstream load, but with a custom ``InferenceSession`` so the thread
    setting takes effect on a newly created session (no installed-package edits).
    """
    from piper import PiperVoice
    from piper.config import PiperConfig
    from piper.phonemize_espeak import ESPEAK_DATA_DIR

    if not model_path.is_file():
        raise FileNotFoundError(f"Piper voice model not found: {model_path}")

    use_cuda = str(os.environ.get("PIPER_USE_CUDA", "")).strip().lower() in ("1", "true", "yes")
    intra = onnx_intra_op_threads()
    model_key = _voice_key(model_path)
    if intra <= 0:
        voice = PiperVoice.load(model_path, use_cuda=use_cuda)
        return _wrap_voice_phonemize_cache(voice, model_key)

    import json

    import onnxruntime

    config_path = Path(str(model_path) + ".json")
    if not config_path.is_file():
        raise FileNotFoundError(f"Piper voice config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)

    if use_cuda:
        providers: list[Any] = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
        ]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = int(intra)
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers,
    )
    print(
        f"[PIPER-POOL] onnx_intra_op_threads={intra} model={model_path.name}",
        flush=True,
    )
    voice = PiperVoice(
        config=PiperConfig.from_dict(config_dict),
        session=session,
        espeak_data_dir=Path(ESPEAK_DATA_DIR),
        download_dir=Path.cwd(),
    )
    return _wrap_voice_phonemize_cache(voice, model_key)


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


def _store_loaded_replica(
    key: str,
    voice: Any,
    request_id: str | None,
) -> list[threading.Thread]:
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
        # First replica is ready — fill remaining slots in the background. Keep
        # the originating request id so startup top-ups are identifiable in logs.
        return _start_loaders_locked(entry, request_id)


def _store_load_error(key: str, error: BaseException) -> None:
    with _POOL_LOCK:
        entry = _MODELS.get(key)
        if entry is None:
            return
        entry.inflight = max(0, entry.inflight - 1)
        if entry.inflight == 0 and len(entry.instances) < entry.target_count:
            entry.error = error
            entry.ready.set()


def _load_voice_replica(model_path: Path, request_id: str | None) -> None:
    key = _voice_key(model_path)
    try:
        voice = _create_piper_voice(model_path)
        topup_threads = _store_loaded_replica(key, voice, request_id)
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


def pool_status() -> dict[str, dict[str, Any]]:
    """Return a thread-safe snapshot of replica targets and loading state."""
    with _POOL_LOCK:
        return {
            key: {
                "model_path": str(entry.model_path),
                "speaker_count": entry.weight,
                "ready": len(entry.instances),
                "target": entry.target_count,
                "inflight": entry.inflight,
                "error": repr(entry.error) if entry.error is not None else None,
            }
            for key, entry in _MODELS.items()
        }


def pool_runtime_metrics() -> dict[str, Any]:
    """
    Instrumentation snapshot: active slots, waiting acquires, per-model status.

    Gauge fields do not change scheduling by themselves; reservation policy uses
    ``first_chunk_reserved_slots`` + ``waiting_first_chunk``.
    """
    with _METRICS_LOCK:
        waiting = int(_waiting_acquires)
        waiting_first = int(_waiting_first_chunk)
        active = int(_checked_out)
    status = pool_status()
    return {
        "active_slots": active,
        "waiting_requests": waiting,
        "waiting_first_chunk": waiting_first,
        "first_chunk_reserved": first_chunk_reserved_slots(),
        "slot_budget": pool_slot_budget(),
        "phoneme_cache": phoneme_cache_stats(),
        "models": status,
    }


def last_call_metrics() -> dict[str, Any]:
    """
    Per-thread metrics from the most recent synthesize_wav_file on this thread.

    Missing fields are omitted (caller should treat as unavailable).
    """
    out: dict[str, Any] = {}
    acquire_ms = getattr(_tls, "last_acquire_wait_ms", None)
    synth_ms = getattr(_tls, "last_synth_ms", None)
    if acquire_ms is not None:
        out["acquire_wait_ms"] = int(acquire_ms)
    if synth_ms is not None:
        out["synth_ms"] = int(synth_ms)
    first = getattr(_tls, "last_first_chunk", None)
    if first is not None:
        out["first_chunk"] = bool(first)
    return out


def _metrics_enter_acquire(*, first_chunk: bool) -> None:
    global _waiting_acquires, _waiting_first_chunk
    with _METRICS_LOCK:
        _waiting_acquires += 1
        if first_chunk:
            _waiting_first_chunk += 1


def _metrics_leave_acquire(*, acquired: bool, first_chunk: bool) -> None:
    global _waiting_acquires, _waiting_first_chunk, _checked_out
    with _METRICS_LOCK:
        _waiting_acquires = max(0, _waiting_acquires - 1)
        if first_chunk:
            _waiting_first_chunk = max(0, _waiting_first_chunk - 1)
        if acquired:
            _checked_out += 1


def _metrics_release_checkout() -> None:
    global _checked_out
    with _METRICS_LOCK:
        _checked_out = max(0, _checked_out - 1)


def wait_until_ready(*, timeout_seconds: float = 300.0) -> dict[str, dict[str, Any]]:
    """Block until every registered model has reached its replica target.

    Startup callers use this to ensure WSGI does not begin accepting requests while
    Piper is still loading. A failed replica or timeout is fatal rather than silently
    moving the loading delay onto the first player turn.
    """
    timeout_seconds = max(0.1, float(timeout_seconds))
    deadline = time.perf_counter() + timeout_seconds

    while True:
        status = pool_status()
        pending = {
            key: item
            for key, item in status.items()
            if int(item["ready"]) < int(item["target"])
        }
        if not pending:
            return status

        failed = {
            key: item
            for key, item in pending.items()
            if item["error"] is not None and int(item["inflight"]) == 0
        }
        if failed:
            details = ", ".join(
                f"{Path(key).name}={item['ready']}/{item['target']} error={item['error']}"
                for key, item in failed.items()
            )
            raise RuntimeError(f"Piper pool warm-up failed: {details}")

        if time.perf_counter() >= deadline:
            details = ", ".join(
                f"{Path(key).name}={item['ready']}/{item['target']} "
                f"inflight={item['inflight']}"
                for key, item in pending.items()
            )
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.1f}s warming Piper pool: {details}"
            )

        time.sleep(0.05)


def warm_voice_models(
    model_paths: list[Path],
    *,
    request_id: str | None = None,
    wait_for_ready: bool = False,
    timeout_seconds: float = 300.0,
) -> dict[str, dict[str, Any]] | None:
    """Register models and warm replicas, optionally waiting for full readiness."""
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
    allocation_plan = ""
    model_count = 0
    with _POOL_LOCK:
        for model_path in unique:
            _register_model_locked(model_path)
        _recompute_targets_locked()
        allocation_plan = "; ".join(
            f"{entry.model_path} speakers={entry.weight} target={entry.target_count}"
            for entry in _MODELS.values()
        )
        model_count = len(_MODELS)
        for entry in _MODELS.values():
            threads.extend(_start_loaders_locked(entry, request_id))

    if request_id == "startup":
        print(
            f"[PIPER-POOL] allocation slot_budget={pool_slot_budget()} "
            f"first_chunk_reserved={first_chunk_reserved_slots()} "
            f"onnx_intra_op_threads={onnx_intra_op_threads()} "
            f"models={model_count} plan={allocation_plan}",
            flush=True,
        )

    for thread in threads:
        thread.start()

    if wait_for_ready:
        return wait_until_ready(timeout_seconds=timeout_seconds)
    return None


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


def _followup_may_take(entry: _ModelEntry) -> bool:
    """
    Best-effort reservation: follow-ups leave free replicas for first-chunk TTFA.

    When ``PIPER_FIRST_CHUNK_RESERVED`` is 0, follow-ups compete equally.
    When first-chunk waiters exist, follow-ups only take if free count exceeds the
    reserved floor. If nobody is waiting for a first chunk, follow-ups may use
    any free replica (avoid idle waste).
    """
    reserved = first_chunk_reserved_slots()
    if reserved <= 0:
        return True
    with _METRICS_LOCK:
        waiting_first = int(_waiting_first_chunk)
    free_n = entry.free.qsize()
    if waiting_first <= 0:
        return free_n > 0
    return free_n > reserved


def _acquire_instance(model_path: Path, *, first_chunk: bool = True) -> _VoiceInstance:
    """
    Check out a warm replica.

    ``first_chunk=True`` (default): TTFA / single-utterance path — may use any free
    replica, including those held in reserve for first audio.
    ``first_chunk=False``: follow-up segments — respect ``PIPER_FIRST_CHUNK_RESERVED``.
    """
    key = _voice_key(model_path)
    deadline = time.perf_counter() + 120.0
    retried_error = False
    wait_started = time.perf_counter()
    acquired = False
    _metrics_enter_acquire(first_chunk=first_chunk)
    try:
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

            if not first_chunk and not _followup_may_take(entry):
                # Yield so first-chunk waiters can drain reserved free replicas.
                time.sleep(min(0.05, remaining))
                continue

            try:
                instance = entry.free.get(timeout=min(0.5, remaining))
                # Re-check reservation after get: if a follow-up raced a first-chunk
                # waiter, put back when the free floor would be violated.
                if not first_chunk and first_chunk_reserved_slots() > 0:
                    with _METRICS_LOCK:
                        waiting_first = int(_waiting_first_chunk)
                    if waiting_first > 0 and entry.free.qsize() < first_chunk_reserved_slots():
                        entry.free.put(instance)
                        time.sleep(min(0.05, remaining))
                        continue
                acquired = True
                _tls.last_acquire_wait_ms = max(
                    0, int(round((time.perf_counter() - wait_started) * 1000))
                )
                _tls.last_first_chunk = bool(first_chunk)
                return instance
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
    finally:
        _metrics_leave_acquire(acquired=acquired, first_chunk=first_chunk)


def _release_instance(instance: _VoiceInstance) -> None:
    try:
        with _POOL_LOCK:
            entry = _MODELS.get(instance.model_key)
            if entry is None:
                return
            if len(entry.instances) > entry.target_count and instance in entry.instances:
                entry.instances.remove(instance)
                return
        entry.free.put(instance)
    finally:
        _metrics_release_checkout()


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
    first_chunk: bool = True,
) -> None:
    """Synthesize text to a WAV file using a pooled in-process Piper voice replica.

    ``first_chunk`` defaults True so preview / single-utterance callers keep TTFA
    priority. Segmented follow-ups should pass ``first_chunk=False``.
    """
    from piper import SynthesisConfig

    _tls.last_acquire_wait_ms = None
    _tls.last_synth_ms = None
    _tls.last_first_chunk = None
    instance = _acquire_instance(model_path, first_chunk=first_chunk)
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

        synth_started = time.perf_counter()
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
        _tls.last_synth_ms = max(
            0, int(round((time.perf_counter() - synth_started) * 1000))
        )
    finally:
        _release_instance(instance)


def clear_pool_for_tests() -> None:
    """Reset module state (tests only)."""
    global _waiting_acquires, _waiting_first_chunk, _checked_out
    with _POOL_LOCK:
        _MODELS.clear()
    with _METRICS_LOCK:
        _waiting_acquires = 0
        _waiting_first_chunk = 0
        _checked_out = 0
    _tls.last_acquire_wait_ms = None
    _tls.last_synth_ms = None
    _tls.last_first_chunk = None
    clear_phoneme_cache_for_tests()


def model_replica_counts_for_tests() -> dict[str, int]:
    """Return live replica counts keyed by model path (tests only)."""
    with _POOL_LOCK:
        return {key: len(entry.instances) for key, entry in _MODELS.items()}


def model_targets_for_tests() -> dict[str, int]:
    """Return target replica counts keyed by model path (tests only)."""
    with _POOL_LOCK:
        return {key: entry.target_count for key, entry in _MODELS.items()}
