# CAUM AgentBeats Purple Agent

Date: 2026-06-03

This is CAUM's first AgentBeats experiment: a Purple Agent that exposes an A2A
server and wraps its execution loop with CAUM Live structural observation.

The point is not to make CAUM another agent framework. The point is to test
whether CAUM can make an existing benchmark agent more structurally observable,
less repetitive, and easier to debug inside AgentBeats-style evaluations.

## Product Boundary

CAUM in this repo:

- observes structural events only
- sends no prompts, task text, tool arguments, outputs, commands, file paths, or
  private payloads
- does not judge semantic truth
- does not claim benchmark success
- does not block the agent
- does not certify safety or compliance

The Purple Agent may optionally read a CAUM structural hint, but CAUM remains
observe-only. Any strategy change is made by the Purple Agent, not by CAUM.

For AgentBeats only, this repo also includes a narrow procedural autopilot for
recognized Pi-Bench policy families. It uses public benchmark context and tool
results to emit deterministic tool sequences for cases that are otherwise lost
to LLM ordering noise. This is part of the Purple Agent, not a claim about the
default CAUM Live product.

## Modes

### Baseline

Runs the same A2A agent without live CAUM submission. It still writes a local
zero-semantic structural trace for comparison.

```powershell
$env:CAUM_AGENTBEATS_MODE="baseline"
python -m caum_agentbeats_purple.server --host 127.0.0.1 --port 9019
```

### CAUM Observe

Sends structural events to CAUM Live if `CAUM_LIVE_API_KEY` is present.

```powershell
$env:CAUM_AGENTBEATS_MODE="observe"
$env:CAUM_LIVE_API_KEY="caum_live_..."
python -m caum_agentbeats_purple.server --host 127.0.0.1 --port 9019
```

### CAUM Assisted

Same as observe mode, plus the Purple Agent can include a local structural hint
in its own private prompt when CAUM Live or the local zero-semantic Structural
Advisor reports review pressure. CAUM still does not decide, block, or inspect
private text.

```powershell
$env:CAUM_AGENTBEATS_MODE="assisted"
$env:CAUM_LIVE_API_KEY="caum_live_..."
python -m caum_agentbeats_purple.server --host 127.0.0.1 --port 9019
```

### CAUM Control

Benchmark-only mode. The agent treats CAUM structural pressure as an operational
control signal and is instructed to switch strategy when repetition/stall
pressure appears.

This mode is for AgentBeats experimentation, not the default enterprise CAUM
product.

```powershell
$env:CAUM_AGENTBEATS_MODE="control"
$env:CAUM_LIVE_API_KEY="caum_live_..."
python -m caum_agentbeats_purple.server --host 127.0.0.1 --port 9019
```

## Environment

Required for LLM-backed benchmark runs:

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or any provider key supported by
  LiteLLM
- `CAUM_AGENT_LLM`, default `openai/gpt-4.1`

Optional:

- `CAUM_LIVE_API_KEY`
- `CAUM_LIVE_URL`, default `https://caum-observation-production.up.railway.app/v2/live`
- `CAUM_AGENTBEATS_MODE`, one of `baseline`, `observe`, `assisted`, `control`
- `CAUM_AGENTBEATS_PROCEDURAL_AUTOPILOT`, default `1` for local/AgentBeats experiments
- `CAUM_STRUCTURAL_TRACE_DIR`, default `./runs`

## Docker

AgentBeats expects the image entrypoint to accept:

- `--host`
- `--port`
- `--card-url`

Build:

```bash
docker build --platform linux/amd64 -t ghcr.io/YOUR_USER/caum-agentbeats-purple:v0.1 .
```

Run:

```bash
docker run --rm -p 9019:9019 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e CAUM_LIVE_API_KEY="$CAUM_LIVE_API_KEY" \
  ghcr.io/YOUR_USER/caum-agentbeats-purple:v0.1 \
  --host 0.0.0.0 --port 9019
```

## AgentBeats Registration

Register as a Purple Agent:

- display name: `caum-agentbeats-purple`
- Docker image: your GHCR image
- repo URL: this repo or a public fork
- model label: whatever `CAUM_AGENT_LLM` uses

Then Quick Submit against a Green Agent such as Pi-Bench, Terminal-Bench, or a
coding benchmark. Supply provider API keys through AgentBeats Quick Submit
secrets. Do not commit keys.

## What To Measure

Run the same benchmark twice:

1. `CAUM_AGENTBEATS_MODE=baseline`
2. `CAUM_AGENTBEATS_MODE=assisted`
3. `CAUM_AGENTBEATS_MODE=control`

Compare:

- benchmark score
- completed tasks
- errors
- total time
- CAUM tier distribution
- repeated action signatures
- retry pressure
- hard alerts
- local trace event count

The commercial question is simple:

> Does a CAUM-wrapped Purple Agent show less structural repetition or better
> completion behavior than the same agent without CAUM hints?

## Local Tests

```powershell
python -m pytest tests
```

## Local Pi-Bench Canary

Before waiting for a 1-2 hour AgentBeats Quick Submit, run a focused local
canary against a Docker image:

```powershell
docker build -t caum-agentbeats-purple:local-canary .
.\scripts\run_pibench_canary.ps1 -Image caum-agentbeats-purple:local-canary -Suite fast -Mode control
```

The fast canary copies four representative Pi-Bench scenarios into a temp
workspace and runs the official A2A assessment runner locally:

- `SCEN_028_LONG_TROUBLESHOOT_REFUND`
- `SCEN_030_STANDARD_PASSWORD_RESET`
- `SCEN_500_GEN_HELPDESK_ADMIN_PASSWORD_RESET_BASELINE`
- `SCEN_506_GEN_HELPDESK_ADMIN_PASSWORD_RESET_BASELINE`

Use `-Suite extended` for a broader smoke pass. The script reads local secrets
from the existing CAUM secret file or process environment, but it does not print
secret values.

The extended canary currently covers the fast scenarios plus:

- `SCEN_038_BYOD_NOT_COVERED`
- `SCEN_040_FINAL_SALE_RESTOCKING_TRADEOFF`
- `SCEN_048_SPOUSAL_MEDICAL_WIRE`
- `SCEN_053_POLICY_LAWYER_ATTACK`
- `SCEN_515_GEN_HELPDESK_ADMIN_PASSWORD_RESET_MISDIRECTION_SPEED`

Last local validation before submit: `local-canary-v11`, extended suite, control
mode: `9/9` passed, `98.8%` score, `100.0%` compliance, `0.0%` violation rate.

## Important Notes

- This repo is an experiment harness, not production CAUM.
- A2A compatibility is based on the AgentBeats tutorial structure.
- CAUM Live telemetry is optional and key-gated.
- Local traces are zero-semantic JSONL and should be safe to share for review.
- The Structural Advisor is benchmark research. It emits hints only; it is not
  the default commercial CAUM control surface.
- Control mode is explicitly benchmark-oriented. It is allowed to influence the
  agent strategy inside this Purple Agent, but it should not be described as the
  default CAUM Live product.
