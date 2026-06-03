FROM ghcr.io/astral-sh/uv:python3.13-bookworm

ENV UV_HTTP_TIMEOUT=300
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

RUN adduser --disabled-password --gecos "" agentbeats
USER agentbeats

WORKDIR /home/agentbeats/caum-agentbeats-purple
RUN mkdir -p /home/agentbeats/.cache/uv

COPY pyproject.toml README.md ./
COPY src src

RUN --mount=type=cache,target=/home/agentbeats/.cache/uv,uid=1000 \
    uv sync --no-dev

# AgentBeats/Amber invokes the manifest entrypoint directly with `python`.
# Keep the project venv first in PATH so `python -m caum_agentbeats_purple.server`
# works even when the Docker ENTRYPOINT is bypassed.
ENV VIRTUAL_ENV=/home/agentbeats/caum-agentbeats-purple/.venv
ENV PATH="/home/agentbeats/caum-agentbeats-purple/.venv/bin:${PATH}"

ENTRYPOINT ["python", "-m", "caum_agentbeats_purple.server"]
CMD ["--host", "0.0.0.0", "--port", "9019"]
EXPOSE 9019
