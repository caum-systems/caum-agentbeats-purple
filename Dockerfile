FROM ghcr.io/astral-sh/uv:python3.13-bookworm

ENV UV_HTTP_TIMEOUT=300
ENV PYTHONUNBUFFERED=1

RUN adduser agentbeats
USER agentbeats

WORKDIR /home/agentbeats/caum-agentbeats-purple
RUN mkdir -p /home/agentbeats/.cache/uv

COPY pyproject.toml README.md ./
COPY src src

RUN --mount=type=cache,target=/home/agentbeats/.cache/uv,uid=1000 \
    uv sync --no-dev

ENTRYPOINT ["uv", "run", "python", "-m", "caum_agentbeats_purple.server"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9019
