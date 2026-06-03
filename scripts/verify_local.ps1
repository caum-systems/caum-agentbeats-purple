$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (!(Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -e .[dev]
}

.\.venv\Scripts\python.exe -m pytest tests

@'
from caum_agentbeats_purple.server import build_agent_card
card = build_agent_card("127.0.0.1", 9019)
assert card.name == "caum_agentbeats_purple"
assert card.version == "0.1.0"
print({"agent_card": card.name, "version": card.version, "url": str(card.url)})
'@ | .\.venv\Scripts\python.exe -

Write-Host "Verification complete."
