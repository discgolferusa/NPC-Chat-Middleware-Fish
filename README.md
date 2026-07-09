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

`server.py` loads `game_api.env` on every start. Shell variables still override the file if you set both.

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
| `/voices` | GET | List Piper `.onnx` models under `VOICES_DIR` (default: parent of default voice model) |
| `/voices/preview` | POST | Synthesize sample WAV via Piper |

Preview request body:

```json
{
  "voice_path": "C:\\AI\\Voices\\en\\en_US\\ryan\\medium\\en_US-ryan-medium.onnx",
  "length_scale": 1.15,
  "text": "Optional sample text"
}
```

Returns `audio/wav` (22050 Hz mono PCM16 after ffmpeg normalization).

Env: `VOICES_DIR` overrides the scan root for `/voices`.

## Related documentation

- Yaml Editor: `docs/MIDDLEWARE_INTEGRATION.md` (Funnel + proxy contract)
- Audio format: 22050 Hz, mono, PCM16 WAV
