# CAUM x AgentBeats Experiment Plan

Date: 2026-06-03

## Question

Can CAUM improve or at least make measurable the structural behavior of an
AgentBeats Purple Agent without reading private task content?

## Hypothesis

CAUM will not directly improve semantic correctness by itself. The likely lift
is structural:

- fewer repeated action signatures
- fewer stalled/retry-heavy runs
- lower time on tasks that are already unlikely to progress
- better debugging evidence when a run fails
- possibly better benchmark score when `assisted` mode helps the model change
  strategy under structural pressure

## Experimental Design

Run the same Purple Agent image in three conditions.

### A. Baseline

```bash
CAUM_AGENTBEATS_MODE=baseline
```

No CAUM Live submission. Local zero-semantic trace only.

### B. Observe

```bash
CAUM_AGENTBEATS_MODE=observe
CAUM_LIVE_API_KEY=...
```

CAUM Live receives structural telemetry and produces receipts. The agent does
not read CAUM hints.

### C. Assisted

```bash
CAUM_AGENTBEATS_MODE=assisted
CAUM_LIVE_API_KEY=...
```

The agent may read a private structural hint if CAUM reports T4/T5 or live
alert. CAUM still does not block or decide.

### D. Control

```bash
CAUM_AGENTBEATS_MODE=control
CAUM_LIVE_API_KEY=...
```

Benchmark-only mode. The Purple Agent treats CAUM structural pressure as an
operational control signal and is instructed to switch strategy when structural
repetition/stall pressure appears. This is the aggressive money/leaderboard
mode; it is not the default enterprise CAUM Live product.

## First Tracks

Recommended order:

1. Pi-Bench / Agent Safety
2. Terminal-Bench-style tasks
3. SWE-Bench Pro / coding track

Reason: safety and terminal/coding tracks are closest to CAUM's structural
value. They involve long-ish execution, policies, tool use, retries, and errors.

## Metrics To Compare

Benchmark-native:

- score
- completed
- errors
- time
- violation/refusal/policy metrics where available

CAUM-native:

- current tier
- public class
- live alert count
- event count
- repeated action signature count
- identity stall run count
- exact/pattern cycle coverage
- retry pressure
- burn without progress
- work yield

## Minimum Useful Result

A strong commercial result is not necessarily "higher benchmark score" on day
one.

Minimum useful result:

> CAUM Observe provides evidence that explains why a Purple Agent loses time,
> repeats actions, or fails a task, while preserving zero-semantic boundaries.

Stronger result:

> CAUM Assisted reduces repetition/time/error rate compared with baseline.

Strongest competitive result:

> CAUM Control improves benchmark score or reduces avoidable stuck trajectories.

Best result:

> CAUM Assisted improves an AgentBeats leaderboard score or places competitively
> while producing public-safe structural receipts.

## Claim Boundaries

Do not claim:

- CAUM makes agents safe
- CAUM prevents violations
- CAUM detects hallucinations
- CAUM guarantees benchmark improvement
- CAUM blocks bad behavior

Allowed benchmark-only framing:

- CAUM Control can influence this Purple Agent's strategy inside benchmarks
- CAUM Control is separate from CAUM Live observe-only enterprise mode

Allowed internal framing:

- CAUM makes structural behavior visible
- CAUM provides zero-semantic receipts for agent execution
- CAUM can be tested as an observe-only wrapper around AgentBeats Purple Agents
- assisted mode is benchmark research, not the default commercial product

## Next Implementation Step

Register or locally run the Purple Agent against Pi-Bench.

If AgentBeats registration requires a public GHCR image:

1. publish this folder to a small public/private GitHub repo
2. build Docker image for `linux/amd64`
3. push to GHCR
4. register Purple Agent
5. run Quick Submit with `CAUM_AGENTBEATS_MODE=baseline`
6. repeat with `observe`
7. repeat with `assisted`

## Evidence To Save

For every run, save:

- AgentBeats result JSON/PR link
- CAUM dashboard session links or session hashes
- local `runs/*.jsonl`
- env mode, model, date, track
- exact Docker image digest

This lets us turn the result into a truthful CAUM validation note later.
