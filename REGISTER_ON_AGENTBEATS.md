# Registering CAUM AgentBeats Purple

## 1. Build Image

```bash
docker build --platform linux/amd64 -t ghcr.io/YOUR_USER/caum-agentbeats-purple:v0.1 .
docker push ghcr.io/YOUR_USER/caum-agentbeats-purple:v0.1
```

## 2. Register Purple Agent

Go to AgentBeats Register Agent and choose Purple.

Fields:

- Display name: `caum-agentbeats-purple`
- Image: `ghcr.io/YOUR_USER/caum-agentbeats-purple:v0.1`
- Repository: repository containing this folder
- Description: `A Purple Agent wrapped with CAUM zero-semantic structural observation.`

## 3. Quick Submit

Use a Green Agent page such as Pi-Bench.

Secrets:

- `OPENAI_API_KEY` or equivalent provider key
- optional `CAUM_LIVE_API_KEY`

Config/env for baseline:

```json
{
  "CAUM_AGENTBEATS_MODE": "baseline",
  "CAUM_AGENT_LLM": "openai/gpt-4.1"
}
```

Config/env for observe:

```json
{
  "CAUM_AGENTBEATS_MODE": "observe",
  "CAUM_AGENT_LLM": "openai/gpt-4.1"
}
```

Config/env for assisted:

```json
{
  "CAUM_AGENTBEATS_MODE": "assisted",
  "CAUM_AGENT_LLM": "openai/gpt-4.1"
}
```

Config/env for control:

```json
{
  "CAUM_AGENTBEATS_MODE": "control",
  "CAUM_AGENT_LLM": "openai/gpt-4.1"
}
```

AgentBeats green agents vary in how they pass participant env/config. If a
track does not pass custom env vars, build three tags instead:

- `caum-agentbeats-purple:baseline`
- `caum-agentbeats-purple:observe`
- `caum-agentbeats-purple:assisted`
- `caum-agentbeats-purple:control`

with `ENV CAUM_AGENTBEATS_MODE=...` baked into each image.

## 4. Save Evidence

For each run:

- leaderboard URL
- result PR/commit URL
- model used
- mode
- Docker image digest
- CAUM session hashes
- local structural traces if available

Do not publish CAUM Live keys or provider API keys.
