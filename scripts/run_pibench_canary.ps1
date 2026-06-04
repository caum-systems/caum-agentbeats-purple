param(
    [string]$PiBenchRoot = "$env:LOCALAPPDATA\Temp\pi-bench",
    [string]$Image = "caum-agentbeats-purple:local-canary",
    [ValidateSet("fast", "extended")]
    [string]$Suite = "fast",
    [ValidateSet("baseline", "observe", "assisted", "control")]
    [string]$Mode = "control",
    [int]$Port = 9019,
    [int]$MaxSteps = 50,
    [int]$Seed = 42,
    [switch]$KeepAgent
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $PiBenchRoot)) {
    throw "Pi-Bench root not found: $PiBenchRoot"
}

$secretFile = "C:\Users\LVNET\OneDrive\Desktop\CAUM\_secrets\caum_codex_observer_live.env"
$workRoot = Join-Path $env:LOCALAPPDATA "Temp\caum_pibench_canary"
$scenarioRoot = Join-Path $workRoot "scenarios"
$envFile = Join-Path $workRoot "runtime.env"
$resultFile = Join-Path $workRoot "results_$Suite.json"
$containerName = "caum-purple-local-canary"

function Ensure-A2AMessageIdCompat {
    param([string]$Root)
    $script = @'
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = [
    root / "src/pi_bench/a2a/purple_adapter.py",
    root / "src/pi_bench/a2a/bootstrap.py",
    root / "src/pi_bench/a2a/user_adapter.py",
]
for path in files:
    text = path.read_text()
    if '"messageId": str(uuid.uuid4())' in text:
        continue
    text = text.replace(
        '"role": "user",\n        "parts"',
        '"role": "user",\n        "messageId": str(uuid.uuid4()),\n        "parts"',
    )
    path.write_text(text)
'@
    $script | python - $Root
}

function Write-EnvFile {
    param([string]$Path)
    $lines = @()
    if (Test-Path -LiteralPath $secretFile) {
        foreach ($line in Get-Content -LiteralPath $secretFile) {
            if ($line -match '^[A-Za-z_][A-Za-z0-9_]*=') {
                $lines += $line
            }
        }
    }
    if ($env:OPENAI_API_KEY) {
        $lines += "OPENAI_API_KEY=$($env:OPENAI_API_KEY)"
    }
    if ($env:CAUM_LIVE_API_KEY) {
        $lines += "CAUM_LIVE_API_KEY=$($env:CAUM_LIVE_API_KEY)"
    }
    $lines += "CAUM_AGENT_LLM=openai/gpt-4.1"
    $lines += "CAUM_AGENTBEATS_MODE=$Mode"
    $lines += "CAUM_AGENT_TEMPERATURE=0.0"
    $lines += "CAUM_AGENTBEATS_PROCEDURAL_AUTOPILOT=1"
    $lines += "CAUM_AGENT_LLM_MAX_ATTEMPTS=4"
    $lines | Set-Content -LiteralPath $Path -Encoding ascii
}

function Copy-Scenario {
    param([string]$RelativePath)
    $src = Join-Path (Join-Path $PiBenchRoot "scenarios") $RelativePath
    if (-not (Test-Path -LiteralPath $src)) {
        throw "Scenario not found: $src"
    }
    $dst = Join-Path $scenarioRoot $RelativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $dst) -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Force
}

function Summarize-Results {
    param([string]$Path)
    $summaryScript = @'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
results = data.get("results", [])
passed = sum(1 for r in results if r.get("all_passed"))
print(f"CANARY SUMMARY: {passed}/{len(results)} passed")
for r in results:
    status = "PASS" if r.get("all_passed") else "FAIL"
    scenario = r.get("scenario_id")
    decision = r.get("canonical_decision")
    failed = [
        (o.get("outcome_id"), o.get("type"), o.get("detail"))
        for o in r.get("outcome_results", [])
        if not o.get("passed")
    ]
    print(f"- {status} {scenario} decision={decision}")
    for outcome_id, kind, detail in failed[:5]:
        print(f"  - {outcome_id} {kind}: {detail}")
    if len(failed) > 5:
        print(f"  - ... {len(failed) - 5} more")
'@
    $summaryScript | python - $Path
}

$fastScenarios = @(
    "helpdesk\scen_500_gen_helpdesk_admin_password_reset_baseline.json",
    "helpdesk\scen_506_gen_helpdesk_admin_password_reset_baseline.json",
    "helpdesk\scen_030_standard_password_reset.json",
    "retail\scen_028_long_troubleshoot_refund.json"
)

$extendedScenarios = $fastScenarios + @(
    "helpdesk\scen_515_gen_helpdesk_admin_password_reset_misdirection_speed.json",
    "helpdesk\scen_038_byod_not_covered.json",
    "retail\scen_040_final_sale_restocking_tradeoff.json",
    "finra\scen_048_spousal_medical_wire.json",
    "finra\scen_053_policy_lawyer_attack.json"
)

$selected = if ($Suite -eq "extended") { $extendedScenarios } else { $fastScenarios }

Remove-Item -LiteralPath $workRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $scenarioRoot -Force | Out-Null
foreach ($scenario in $selected) {
    Copy-Scenario $scenario
}
Write-EnvFile $envFile
Ensure-A2AMessageIdCompat $PiBenchRoot

$existing = docker ps -a --filter "name=^/$containerName$" --format "{{.ID}}"
if ($existing) {
    docker rm -f $containerName | Out-Null
}

Write-Output "Starting local CAUM Purple image: $Image"
$containerId = docker run -d --name $containerName --env-file $envFile -p "127.0.0.1:$Port`:9019" $Image
Start-Sleep -Seconds 8

try {
    $null = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/.well-known/agent-card.json" -UseBasicParsing -TimeoutSec 15
    Write-Output "Running Pi-Bench $Suite canary with $($selected.Count) scenarios..."
    docker run --rm --env-file $envFile -e UV_LINK_MODE=copy `
        -v "${PiBenchRoot}:/work" `
        -v "${workRoot}:/canary" `
        -w /work ghcr.io/astral-sh/uv:python3.13-bookworm `
        uv run python examples/a2a_demo/run_a2a.py `
        --external `
        --host host.docker.internal `
        --port $Port `
        --scenarios-dir /canary/scenarios `
        --user-model gpt-4.1-mini `
        --max-steps $MaxSteps `
        --concurrency 1 `
        --seed $Seed `
        --save-to "/canary/$(Split-Path -Leaf $resultFile)"
    Summarize-Results $resultFile
    Write-Output "Saved full result: $resultFile"
}
finally {
    if (-not $KeepAgent) {
        docker rm -f $containerName | Out-Null
    }
}
