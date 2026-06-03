param(
    [string]$Mode = "baseline",
    [int]$Port = 9019,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (!(Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -e .[dev]
}

$env:CAUM_AGENTBEATS_MODE = $Mode
$env:CAUM_STRUCTURAL_TRACE_DIR = Join-Path $root "runs"

Write-Host "Starting CAUM AgentBeats Purple Agent on http://$HostName`:$Port/ in mode=$Mode"
.\.venv\Scripts\python.exe -m caum_agentbeats_purple.server --host $HostName --port $Port
