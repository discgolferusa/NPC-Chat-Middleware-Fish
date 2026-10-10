# Piper concurrency investigation (findings)

Date: 2026-10-09  
Scope: read-only analysis of `piper_pool.py`, `tts_processing.py`, `server.synthesize_speech`, and upstream `piper`/`onnxruntime`. **No production lock/allocation changes.**

## Executive summary

Our **pool code does not hold `_POOL_LOCK` during ONNX inference**. Different checked-out replicas *can* call `session.run` from different threads.

However, **upstream Piper serializes phonemization** with a **process-wide** `_ESPEAK_PHONEMIZER_LOCK`, and each replica’s ONNX session is created with **default `SessionOptions()`** (`intra_op_num_threads=0` = auto = typically all cores). Together those explain “8 replicas but concurrent latency collapses toward serial”:

1. Phonemize phases cannot overlap (hard lock).
2. Concurrent `session.run` calls likely **oversubscribe CPU** (N sessions × all cores), so wall time rises almost linearly even when inference intervals technically overlap.

## 1. Locks / sync involved in acquire + use

| Sync primitive | Location | Held during |
|----------------|----------|-------------|
| `_POOL_LOCK` | [`piper_pool.py`](../piper_pool.py) | Register models, recompute targets, start loaders, snapshot acquire state, trim surplus, store replica, status |
| `entry.free` (`queue.Queue`) | same | Blocking wait for a free `_VoiceInstance` (timeout 0.5s loops) |
| `entry.ready` (`threading.Event`) | same | Wait for first replica load (outside `_POOL_LOCK`) |
| `_METRICS_LOCK` | same | Instrumentation gauges only |
| **`_ESPEAK_PHONEMIZER_LOCK`** | **`piper/voice.py` (site-packages)** | **Entire espeak phonemize of each text part** |
| Global `_ESPEAK_PHONEMIZER` singleton | `piper/voice.py` | Shared across all `PiperVoice` instances |
| Native `espeakbridge` | `piper/phonemize_espeak.py` | `set_voice` + `get_phonemes` under that lock |

`tts_processing.py` has **no locks**; it only reads `.onnx.json` metadata and builds FFmpeg filters.

## 2. Pool bookkeeping released before expensive work?

**Yes for our pool.**

Acquire path ([`_acquire_instance`](../piper_pool.py)):

- `_kick_loaders` / `_snapshot_acquire_state` take `_POOL_LOCK` only briefly.
- Blocking waits use `entry.free.get(...)` and `ready_event.wait(...)` **outside** `_POOL_LOCK`.
- On success, returns a `_VoiceInstance`; lock is not held.

Synthesis path ([`synthesize_wav_file`](../piper_pool.py)):

```text
instance = _acquire_instance(...)   # lock not held after return
voice.synthesize(...)               # ONNX + phonemize here
_release_instance(instance)         # brief lock only if trimming surplus
```

Release puts the instance back on `free` **after** releasing `_POOL_LOCK` when not trimming (`entry.free.put` outside the `with` block).

## 3. Can different instances synthesize concurrently?

**At the pool layer: yes.** Distinct replicas are distinct `PiperVoice` objects, each with its own `onnxruntime.InferenceSession` (`PiperVoice.load` → new session). Checked-out instances are removed from `free`, so two threads get different objects when ≥2 replicas are free.

**At the Piper phonemize layer: no overlapping phonemize.** All ESPEAK voices share one lock.

**At ONNX Runtime: possible concurrent `session.run`, but CPU contention may erase speedup.**

## 4. Shared global / singleton / subprocess serialization?

| Mechanism | Serializes? |
|-----------|-------------|
| `_ESPEAK_PHONEMIZER` + `_ESPEAK_PHONEMIZER_LOCK` | **Yes — phonemize** |
| One `InferenceSession` per replica | No shared session across replicas |
| `PIPER_USE_SUBPROCESS` | If enabled, each synth is a separate OS process (not default in-process path) |
| Flask `threaded=True` | Allows concurrent request threads (does not serialize TTS by itself) |
| FFmpeg in `convert_wav_to_standard_format` | Separate subprocess per clip **after** Piper; can add post-TTS contention but is outside the pool |

## 5. Model load once vs reopen every synth?

**Loaded once per replica at warm/load time** via `_create_piper_voice` → `PiperVoice.load` → `InferenceSession(str(model_path), ...)`.

Each synth uses the in-memory session (`self.session.run`); it does **not** reopen the `.onnx` file per utterance.

`load_piper_model_meta` only reads JSON sidecars for speaker counts / defaults (pool weighting), not for each synth.

## 6. ONNX Runtime threading per instance

From `PiperVoice.load` (installed `piper/voice.py`):

```python
session=onnxruntime.InferenceSession(
    str(model_path),
    sess_options=onnxruntime.SessionOptions(),  # defaults
    providers=providers,
)
```

Observed defaults on this toolchain:

- `intra_op_num_threads = 0` → **auto** (typically ≈ CPU count)
- `inter_op_num_threads = 0` → auto
- `execution_mode = ORT_SEQUENTIAL`

We do **not** set per-session thread caps. With 8 LibriTTS replicas, concurrent synth can launch many ORT threads → oversubscription.

## LibriTTS High × 8 replicas

Pool weighting uses `num_speakers` from `.onnx.json` (`allocate_replicas`). A high-speaker LibriTTS model can claim most/all of `PIPER_POOL_SLOTS` (default 10), matching “eight loaded instances” when other models take remaining slots or budget is 8.

## Diagnostic benchmark

Script (no production lock changes):

[`scripts/diagnose_piper_concurrency.py`](../scripts/diagnose_piper_concurrency.py)

Records per request:

- `request_id`, `piper_instance_id` (`session:<id(session)>`)
- acquire start/end, phonemize start/end, inference start/end
- `thread_id`, `process_id`

Run on the Ryzen host (model path required):

```powershell
cd C:\path\to\NPC-Chat-Middleware
$env:PIPER_POOL_SLOTS = "8"
$env:PIPER_DIAG_MODEL = "C:\path\to\en_US-libritts-high.onnx"
python scripts\diagnose_piper_concurrency.py --concurrency 4 --json-out piper-diag.json
```

Interpretation:

| Observation | Meaning |
|-------------|---------|
| `phonemize_overlap_pair_count == 0` | Expected — global espeak lock |
| `inference_overlap_pair_count > 0` | Pool allows concurrent `session.run` |
| `inference_overlap_pair_count == 0` but distinct instances > 1 | Contended/queued at ORT/CPU or unexpected serialization |
| Wall ≈ sum of singles | Effective serialization (lock and/or CPU oversubscription) |
| Wall ≈ max of singles | True parallel speedup |

## Recommended next steps (not done yet)

1. Run the diagnostic on Ryzen with LibriTTS High; attach `piper-diag.json`.
2. If phonemize is a large fraction of wall time: consider espeak isolation (process pool) or phonemize once / cache — **separate design**.
3. If inference overlaps but wall still ~serial: cap `SessionOptions.intra_op_num_threads` per replica (e.g. 1–2) so N concurrent sessions fit the machine.
4. Do **not** change pool locking until diag confirms the bottleneck.
