# Starts Tailscale Funnel (if needed) and middleware with the local venv.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-GameApiPort {
    if ($env:GAME_API_PORT -match '^\d+$') {
        return [int]$env:GAME_API_PORT
    }

    $envFile = Join-Path $Root "game_api.env"
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            $trimmed = $line.Trim()
            if ($trimmed -and -not $trimmed.StartsWith("#") -and $trimmed -match '^GAME_API_PORT\s*=\s*(\d+)') {
                return [int]$Matches[1]
            }
        }
    }

    return 5000
}

function Ensure-TailscaleFunnel {
    param([int]$Port)

    $tailscale = Get-Command tailscale -ErrorAction SilentlyContinue
    if (-not $tailscale) {
        Write-Warning "[FUNNEL] tailscale not found on PATH; skipping funnel setup."
        return
    }

    $tailscaleStatus = & tailscale status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[FUNNEL] tailscale is not running; start Tailscale app first."
        return
    }

    $funnelStatus = (& tailscale funnel status 2>&1 | Out-String).Trim()
    if ($funnelStatus -match "127\.0\.0\.1:$Port\b" -or $funnelStatus -match "localhost:$Port\b") {
        Write-Host "[FUNNEL] already exposing localhost:$Port"
        Write-Host $funnelStatus
        return
    }

    Write-Host "[FUNNEL] starting tailscale funnel for localhost:$Port ..."
    & tailscale funnel --bg $Port
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[FUNNEL] tailscale funnel failed (exit $LASTEXITCODE). Middleware will still start locally."
        return
    }

    Write-Host "[FUNNEL] public URL:"
    & tailscale funnel status
}

$venvCandidates = @(
    (Join-Path $Root ".venv\Scripts\Activate.ps1"),
    "C:\AI\npc-server\.venv\Scripts\Activate.ps1"
)

$activated = $false
foreach ($candidate in $venvCandidates) {
    if (Test-Path $candidate) {
        . $candidate
        $activated = $true
        break
    }
}

if (-not $activated) {
    Write-Error "No venv found. Create .venv in this folder or at C:\AI\npc-server\.venv"
}

Set-Location $Root

$gameApiPort = Get-GameApiPort
Ensure-TailscaleFunnel -Port $gameApiPort

python server.py
