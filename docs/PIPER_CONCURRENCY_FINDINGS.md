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

Opt-in cap: set `PIPER_ONNX_INTRA_OP_THREADS` (default unset/`0` = stock automatic).
When `>= 1`, [`piper_pool._create_piper_voice`](../piper_pool.py) builds the
`InferenceSession` with `SessionOptions.intra_op_num_threads` set — `PiperVoice.load()`
does not expose session options, so we mirror its constructor path without editing
site-packages. Production default remains unchanged (automatic ORT threading).

## LibriTTS High × 8 replicas

Pool weighting uses `num_speakers` from `.onnx.json` (`allocate_replicas`). A high-speaker LibriTTS model can claim most/all of `PIPER_POOL_SLOTS` (default 10), matching “eight loaded instances” when other models take remaining slots or budget is 8.

**P1 — explicit budgets:** set `PIPER_MODEL_SLOT_BUDGETS=en_US-libritts-high:8,lessac:2`
(stem/substring match) to stop LibriTTS speaker-count from starving narrator/Clean voices.

**P1 — TTFA-fast FFmpeg:** first streaming clip uses a lighter `-af` chain when
`PIPER_TTFA_FAST_FFMPEG` is unset/`1` (skip afftdn/dynaudnorm/full reverse trim).
Follow-up clips keep the full clarity chain. Set `0` to disable.

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

## ORT intra-op threading matrix

Script (pool allocation / speaker / synth logic unchanged; rebuilds sessions per config):

[`scripts/benchmark_piper_ort_threads.py`](../scripts/benchmark_piper_ort_threads.py)

Uses Platform fixtures `perf/fixtures/dialogue_{short,medium,long}.txt` and fail-closed
safety from `NPC-Platform/perf` (no OpenRouter / Discord).

```powershell
cd C:\Users\jayhi\source\repos\NPC-Chat-Middleware
.\.venv\Scripts\python.exe scripts\benchmark_piper_ort_threads.py `
  --model C:\AI\Voices\en_US-libritts-high.onnx `
  --platform-root C:\Users\jayhi\source\repos\NPC-Platform `
  --slots 8 --repeats 3 `
  --json-out reports\ort-threads.json
```

## Recommended next steps

1. Run the ORT thread matrix on Ryzen; review `ort-threads.md` recommendations for c=1/4/8.
2. If a capped setting wins under concurrency, set `PIPER_ONNX_INTRA_OP_THREADS` on the host (optional; default 0 preserves today’s behavior).
3. If phonemize remains a large fraction of wall: consider espeak isolation — **separate design**.
4. Do **not** change pool locking; diag already confirmed overlapping inference.
