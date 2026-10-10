# S1 v2 — Production-equivalent Piper TTS / server-side TTFA

## Purpose

Measure **server-side time to first audio (TTFA)** and complete response synthesis using the **same segmentation and clip pipeline** as production voice turns (`/npc-voice-turn` with `stream_clips`), without STT, LLM, OpenRouter, or Discord.

Primary metric:

`server_ttfa_ms = first_audio_ready - tts_request_start`

where `first_audio_ready` is after clip-0 Piper synthesis **and** ffmpeg packaging (same moment as `note_first_audio_ms`).

This is **not** browser playback-start latency.

## Production path (faithfulness)

```text
npc_text
  → split_into_ordered_tts_segments
       → clean_for_tts → split_text_for_streaming (140 / 320 char budgets)
  → synthesize_segmented_speech_clips(..., return_after_first_clip=True)
       → clip 0 on caller thread (Piper → ffmpeg)
       → remaining clips sequential on background thread
```

- Piper instances are acquired **per segment**, not held for the whole response.
- Within one response, clips never synthesize concurrently.
- Optional `on_timing_event` on `synthesize_segmented_speech_clips` is benchmark-only; omitting it leaves production unchanged.

S1 v1 (`/voices/preview` full-utterance) remains separate for Piper pool regression.

## Architecture

| Piece | Location |
|-------|----------|
| Timing hooks | `server.synthesize_segmented_speech_clips(on_timing_event=...)` |
| Facade | `tts_benchmark_pipeline.run_production_tts_pipeline` |
| Workload helpers | `s1v2_workload.py` |
| CLI runner | `scripts/run_s1v2_ttfa.py` |
| Platform orchestrator | `NPC-Platform/perf` → `python -m perf s1v2` |

## Scenarios

| Flag | Meaning |
|------|---------|
| `a` | c=1 per dialogue class |
| `b` / `ab` | Burst ladder 1–10 concurrent **responses** |
| `c` / `arrival` | Seeded Poisson / burst / sustained arrivals |
| `d` | Mixed lengths 40/50/10 (+ arrivals) |
| `ort` / `--ort-threads 0,2,4` | Fresh ORT sessions per setting |
| `all` | Combined |

## Windows Ryzen (example)

```powershell
cd C:\Users\jayhi\source\repos\NPC-Chat-Middleware
.\.venv\Scripts\pip install psutil
.\.venv\Scripts\python.exe scripts\run_s1v2_ttfa.py `
  --voice C:\AI\Voices\en\en_US\Licensed\libritts\en_US-libritts-high.onnx `
  --platform-root C:\Users\jayhi\source\repos\NPC-Platform `
  --scenario ab `
  --iterations 8 `
  --ort-threads 0,2,4 `
  --reports-dir C:\Users\jayhi\source\repos\NPC-Platform\perf\reports
```

Or via platform:

```powershell
cd C:\Users\jayhi\source\repos\NPC-Platform
$env:PERF_VOICE_PATH = "C:\AI\Voices\en\en_US\Licensed\libritts\en_US-libritts-high.onnx"
$env:NPC_MIDDLEWARE_ROOT = "C:\Users\jayhi\source\repos\NPC-Chat-Middleware"
python -m perf s1v2 --scenario ab --iterations 8 --ort-threads 0,2,4
```

Dual models:

```powershell
.\.venv\Scripts\python.exe scripts\run_s1v2_ttfa.py `
  --voice C:\path\to\en_US-libritts-high.onnx `
  --secondary-voice C:\path\to\en_US-lessac-medium.onnx `
  --model-mode both `
  --speaker-ids 0,12,40,100 `
  --scenario all `
  --platform-root C:\Users\jayhi\source\repos\NPC-Platform
```

## Linux / DigitalOcean notes

- Use middleware `.venv/bin/python` and the same CLI flags.
- Ensure `ffmpeg` is on `PATH` (production post-process runs per clip).
- Set `VOICES_DIR` / absolute `--voice` paths inside the container or host mount.
- `psutil` is required for CPU/RSS; without it those fields are `"unavailable"`.
- Fail-closed allowlist still applies via platform `perf/config`.

## Outputs

Under `perf/reports/<env>/s1v2/<timestamp>/`:

- `report.json`, `responses.csv`, `cells.csv`
- `capacity_summary.json` (`measured` vs `assumed` vs `extrapolated`)
- `summary.md`

## Capacity caveat

Do **not** equate Piper replica count with table count. Extrapolated tables use measured responses/s and configurable assumptions in `perf/config/capacity_assumptions.yaml`.

## Safety

- No OpenRouter / Discord
- Local allowlisted harness config
- Concurrency and duration caps
- Pool cleared between ORT configurations
- Temp `OUTPUT_DIR` for WAV artifacts
