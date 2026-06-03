from __future__ import annotations

import argparse

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from .executor import Executor


def build_agent_card(host: str, port: int, card_url: str | None = None) -> AgentCard:
    skill = AgentSkill(
        id="benchmark_task_fulfillment",
        name="Benchmark Task Fulfillment",
        description="Solves AgentBeats tasks while emitting CAUM zero-semantic structural telemetry.",
        tags=["agentbeats", "purple-agent", "caum", "structural-observation"],
        examples=[],
    )
    return AgentCard(
        name="caum_agentbeats_purple",
        description="AgentBeats Purple Agent wrapped with CAUM structural observation.",
        url=card_url or f"http://{host}:{port}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CAUM AgentBeats Purple Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    request_handler = DefaultRequestHandler(agent_executor=Executor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=build_agent_card(args.host, args.port, args.card_url), http_handler=request_handler)
    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()
