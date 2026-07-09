# Starts middleware with the venv next to this script (or C:\AI\npc-server\.venv).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

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
python server.py
