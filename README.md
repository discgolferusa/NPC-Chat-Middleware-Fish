# NPC-Chat-Middleware

Python Flask game API for NPC voice turns, DM channel, campaigns, and LLM routing. **Browsers and Discord never call this service directly** — [NPC-Yaml-Editor-CSharp](https://github.com/discgolferusa/NPC-Yaml-Editor-CSharp) proxies all game traffic via `GameProxyService`.

## Local development

```powershell
python server.py
```

Defaults: `127.0.0.1:5000`, no API key required.

Pi and other tailnet devices can use the Ryzen Tailscale IP directly:

```text
http://100.x.x.x:5000
```

Set that address in Yaml Editor `GameApi:BaseUrl` for local testing.

## Alpha: Tailscale Funnel (SmarterASP → home server)

SmarterASP cannot join your tailnet. Expose middleware to the editor host only via **Tailscale Funnel**, with middleware bound to localhost and a shared API key.

### Architecture

```text
Browser ──HTTPS──► SmarterASP (Yaml Editor)
                        │
                        │  Authorization: Bearer <GAME_API_KEY>
                        ▼
              https://<funnel-host>.ts.net
                        │
                        ▼
              127.0.0.1:5000  (this server)
```

### 1. Environment variables (Ryzen)

Generate a long random secret and save it in **`game_api.env`** next to `server.py` (copy from `game_api.env.example` — **never commit** `game_api.env`):

| Variable | Example | Purpose |
|----------|---------|---------|
| `GAME_API_HOST` | `127.0.0.1` | Listen on localhost only |
| `GAME_API_PORT` | `5000` | Port |
| `GAME_API_KEY` | `<random-secret>` | Require Bearer token on game routes |
| `REQUIRE_GAME_API_KEY` | `true` | Block `/llm/reset-usage` when funneled |

```powershell
copy game_api.env.example game_api.env
# Edit game_api.env — set GAME_API_KEY to a long random value

.\start-middleware.ps1
```

`start-middleware.ps1` activates the venv, ensures Tailscale Funnel is running for `GAME_API_PORT` (default 5000), then starts `server.py`.

`server.py` loads `game_api.env` on every start. For `GAME_API_*` keys, **the file wins** over a stale shell export (so editing `game_api.env` and restarting always applies). Other keys still leave existing shell values alone.

One-time alternative (lost when you close PowerShell):

```powershell
$env:GAME_API_HOST = "127.0.0.1"
$env:GAME_API_KEY = "<your-secret>"
$env:REQUIRE_GAME_API_KEY = "true"
python server.py
```

### 2. Tailscale Funnel

Confirm Tailscale is running:

```powershell
tailscale status
```

Expose localhost middleware over HTTPS:

```powershell
tailscale funnel --bg 5000
```

Or with explicit HTTPS port:

```powershell
tailscale funnel --bg --https=443 127.0.0.1:5000
```

Record the public URL:

```powershell
tailscale funnel status
```

Example: `https://ryzen-game.<tailnet>.ts.net`

To tear down:

```powershell
tailscale funnel reset
```

### 3. Yaml Editor (SmarterASP)

Set in the host control panel or production config (do not commit the key):

```json
"GameApi": {
  "BaseUrl": "https://ryzen-game.<tailnet>.ts.net",
  "ApiKey": "<same as GAME_API_KEY>"
}
```

### 4. Smoke tests

From a device **not** on the tailnet (e.g. phone on cellular):

```bash
# Health is public (no auth)
curl https://<funnel-host>/health

# Game routes require auth
curl https://<funnel-host>/session/status
# → 401 Unauthorized

curl -H "Authorization: Bearer <GAME_API_KEY>" https://<funnel-host>/session/status
# → 200 JSON
```

On the live SmarterASP site: sign in → DM console → session status and campaign select should load (not 502).

### 5. Security notes

- `/health` is unauthenticated for sanity checks; it only returns route metadata.
- When `REQUIRE_GAME_API_KEY=true`, `/llm/reset-usage` returns 404 (LAN-only maintenance).
- Rejected auth attempts are logged without the provided key.
- Rotate `GAME_API_KEY` if the Funnel URL leaks.
- Optional: restrict Funnel in Tailscale ACL admin.

## Docker (all-in-one VPS)

Production all-in-one layout (editor + this middleware + Discord bot) lives in the
**NPC-Platform** workspace under `deploy/` (Compose + bootstrap scripts).

This repo provides a `Dockerfile` and `requirements.txt`. Piper resolution order:

1. `PIPER_EXE` env
2. `piper` on `PATH`
3. Legacy Windows `C:\AI\npc-server\.venv\Scripts\piper.exe` (local Funnel/dev)

Local `start-middleware.ps1` + Funnel workflow is unchanged.

## Authentication

When `GAME_API_KEY` is set, all routes except `/health` require:

```http
Authorization: Bearer <GAME_API_KEY>
```

Alternative header (same value):

```http
X-Game-Api-Key: <GAME_API_KEY>
```

## Voice preview (Yaml Editor)

| Route | Method | Purpose |
|-------|--------|---------|
| `/voices` | GET | List Piper `.onnx` models under `VOICES_DIR` (recursive scan; includes dataset, quality, and speaker metadata) |
| `/voices/preview` | POST | Synthesize sample WAV via Piper + FFmpeg clarity/FX chain |

Preview request body:

```json
{
  "voice_path": "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
  "length_scale": 1.15,
  "noise_scale": 0.667,
  "noise_w": 0.8,
  "sentence_silence": 0.2,
  "speaker_id": 0,
  "pitch_semitones": 0,
  "fx_preset": "none",
  "text": "Optional sample text"
}
```

`voice_path` / YAML `voice_model` must be **relative to `VOICES_DIR`** (posix separators). Absolute/UNC paths are rejected; legacy absolutes may still resolve by basename via the voice inventory.

`fx_preset` values: `none`, `clarity`, `cave`, `radio`, `grit`, `underwater`, `robot`, `whisper`.

Returns `audio/wav` (22050 Hz mono PCM16 after ffmpeg loudness/EQ normalization).

NPC YAML TTS fields: `voice_model`, `tts_length_scale`, `tts_noise_scale`, `tts_noise_w`,
`tts_sentence_silence`, `tts_speaker_id`, `tts_pitch_semitones`, `tts_fx`.

Narrator TTS (`ServerSettings/tts.yaml` → `narrator:`) uses the same knobs with unprefixed
names: `voice_model`, `length_scale`, `noise_scale`, `noise_w`, `sentence_silence`,
`speaker_id`, `pitch_semitones`, `fx_preset`. There is no `fallback_voice_model`.

Env: `VOICES_DIR` overrides the scan root for `/voices`. By default the server scans **`../Voices`** (sibling of the middleware repo folder), then falls back to the `Voices` segment in the default Ryan model path (e.g. `C:\AI\Voices`).

### Streaming voice turns

Browser player turns may send form field ``stream_clips=1``. Middleware then returns HTTP after the **first** Piper clip (header ``X-Turn-Streaming: 1``) and finishes remaining clips in the background while ``/turn/progress`` publishes them for SignalR. Without the flag, behavior is unchanged (wait for all clips).

### Speaker annotation sidecars

Optional file beside a multi-speaker model: ``<model>.onnx.speakers.json``.

Example path: `.../Clean 100/medium/clean100.onnx.speakers.json`

```json
{
  "version": 1,
  "speakers": {
    "0": { "name": "Benjamin Parker", "annotation": "masculine, calm, kind" },
    "1": "feminine, soft, sincere"
  }
}
```

Middleware merges `annotation` onto `/voices` → `speakers[]` **only for ids already in the model** (`speaker_id_map` / `num_speakers`). Sidecar ids that Piper cannot synthesize are ignored. The Yaml Editor dropdown shows `Name (annotation)`.

Build from a LibriTTS-P style XLSX (Annotator 3 → 1 → 2 priority):

```bash
python scripts/build_speaker_annotation_sidecar.py --xlsx clean100_voice_descriptors.xlsx --onnx-json "Z:/Voices/.../clean100.onnx.json"
```

## Related documentation

- Yaml Editor: `docs/MIDDLEWARE_INTEGRATION.md` (Funnel + proxy contract)
- Audio format: 22050 Hz, mono, PCM16 WAV
