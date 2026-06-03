from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
import litellm

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from .action_format import parse_action
from .caum_layer import CaumStructuralObserver, stable_hash


load_dotenv()


SYSTEM_PROMPT = """
You are a benchmark participant agent exposed through AgentBeats A2A.

Return exactly one JSON object. If the benchmark expects an action, use:
{"name": "respond", "arguments": {"content": "..."}}

Follow the benchmark instructions and policies. Do not mention CAUM unless the
task explicitly asks about instrumentation. CAUM structural hints are private
execution telemetry, not task content.
""".strip()


PI_BENCH_SYSTEM_PROMPT = """
You are a Pi-Bench policy agent exposed through AgentBeats A2A.

Use the provided tool schemas when available. When a final benchmark decision
is required and the record_decision tool is available, call record_decision
with one of: ALLOW, ALLOW-CONDITIONAL, DENY, ESCALATE.

Follow the benchmark context exactly. Do not mention CAUM unless the task
explicitly asks about instrumentation. CAUM structural hints are private
execution telemetry, not task content.
""".strip()


def extract_data_payload(message: Message) -> tuple[dict[str, Any] | None, str]:
    text_chunks: list[str] = []
    data_obj: Any = None
    for part in message.parts:
        root = part.root
        if isinstance(root, DataPart) and data_obj is None:
            data_obj = root.data
        elif isinstance(root, TextPart):
            text_chunks.append(root.text)
    payload = data_obj if isinstance(data_obj, dict) else None
    return payload, "\n".join(text_chunks)


def benchmark_context_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[str] = []
    for node in payload.get("benchmark_context") or []:
        if not isinstance(node, dict):
            continue
        content = str(node.get("content", "")).strip()
        if not content:
            continue
        kind = str(node.get("kind", "context")).strip() or "context"
        title = kind.replace("_", " ").title()
        blocks.append(f"### {title}\n{content}")
    if not blocks:
        return [{"role": "system", "content": PI_BENCH_SYSTEM_PROMPT}]
    return [
        {
            "role": "system",
            "content": PI_BENCH_SYSTEM_PROMPT + "\n\n## Benchmark Context\n" + "\n\n".join(blocks),
        }
    ]


def format_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        function = getattr(tc, "function", None)
        formatted.append(
            {
                "id": getattr(tc, "id", ""),
                "name": getattr(function, "name", "unknown"),
                "arguments": getattr(function, "arguments", "{}"),
            }
        )
    return formatted


class Agent:
    def __init__(self, observer: CaumStructuralObserver | None = None):
        self.model = os.getenv("CAUM_AGENT_LLM", os.getenv("TAU2_AGENT_LLM", "openai/gpt-4.1"))
        self.observer = observer or CaumStructuralObserver()
        self.messages: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload, raw_text = extract_data_payload(message)
        if payload is not None and isinstance(payload.get("messages"), list):
            await self._run_pi_bench(payload, updater)
            return

        input_text = get_message_text(message)
        if not input_text and raw_text:
            input_text = raw_text
        task_hash = stable_hash(input_text, "task")
        self.observer.observe("task_received", phase="receive", tool="a2a_message", state=task_hash)

        await updater.update_status(TaskState.working, new_agent_text_message("Working..."))

        hint = self.observer.hint()
        user_content = input_text
        if hint.hint:
            user_content = f"{input_text}\n\n[Private structural execution hint]\n{hint.hint}"
            self.observer.observe("structural_hint_used", phase="plan", tool="caum_hint", state={"tier": hint.tier})

        self.messages.append({"role": "user", "content": user_content})

        started = time.perf_counter()
        self.observer.observe("llm_call_started", phase="model", tool="litellm", state={"model": self.model})
        try:
            completion = litellm.completion(
                model=self.model,
                messages=self.messages,
                temperature=float(os.getenv("CAUM_AGENT_TEMPERATURE", "0.0")),
                response_format={"type": "json_object"},
            )
            assistant_content = completion.choices[0].message.content or "{}"
            usage = getattr(completion, "usage", None)
            tokens_used = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else None
            latency_ms = int((time.perf_counter() - started) * 1000)
            self.observer.observe(
                "llm_call_completed",
                phase="model",
                tool="litellm",
                state={"model": self.model, "response_shape": "json_object"},
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )
            assistant_json = parse_action(assistant_content)
        except Exception as exc:
            self.observer.observe("llm_call_error", phase="model", tool="litellm", status="error", state={"error": type(exc).__name__})
            assistant_json = {
                "name": "respond",
                "arguments": {"content": "I ran into an execution error while processing the task."},
            }
            assistant_content = json.dumps(assistant_json, sort_keys=True)

        self.messages.append({"role": "assistant", "content": assistant_content})
        action_name = str(assistant_json.get("name") or "respond")
        self.observer.observe("action_emitted", phase="respond", tool="agent_action", state={"action": action_name})

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=assistant_json))],
            name="Action",
        )
        self.observer.observe("task_artifact_added", phase="respond", tool="a2a_artifact", state={"action": action_name})

    async def _run_pi_bench(self, payload: dict[str, Any], updater: TaskUpdater) -> None:
        messages = list(payload.get("messages") or [])
        tools = list(payload.get("tools") or [])
        seed = payload.get("seed")

        self.observer.observe(
            "pi_bench_turn_received",
            phase="receive",
            tool="a2a_data_part",
            state={
                "messages": len(messages),
                "tools": len(tools),
                "has_seed": seed is not None,
            },
        )

        await updater.update_status(TaskState.working, new_agent_text_message("Working..."))

        hint = self.observer.hint()
        llm_messages = benchmark_context_messages(payload)
        if hint.hint:
            llm_messages.append({"role": "system", "content": f"Private structural execution hint:\n{hint.hint}"})
            self.observer.observe("structural_hint_used", phase="plan", tool="caum_hint", state={"tier": hint.tier})
        llm_messages.extend(m for m in messages if isinstance(m, dict))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": llm_messages,
            "temperature": float(os.getenv("CAUM_AGENT_TEMPERATURE", "0.0")),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if seed is not None:
            kwargs["seed"] = seed

        started = time.perf_counter()
        self.observer.observe("llm_call_started", phase="model", tool="litellm", state={"model": self.model, "mode": "pi_bench"})
        try:
            completion = litellm.completion(**kwargs)
            msg = completion.choices[0].message
            content = getattr(msg, "content", None)
            tool_calls = format_tool_calls(getattr(msg, "tool_calls", None))
            usage = getattr(completion, "usage", None)
            tokens_used = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else None
            latency_ms = int((time.perf_counter() - started) * 1000)
            self.observer.observe(
                "llm_call_completed",
                phase="model",
                tool="litellm",
                state={"model": self.model, "response_shape": "pi_bench", "tool_calls": len(tool_calls)},
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            self.observer.observe("llm_call_error", phase="model", tool="litellm", status="error", state={"error": type(exc).__name__})
            content = f"Agent error: {type(exc).__name__}"
            tool_calls = []

        data: dict[str, Any] = {}
        if tool_calls:
            data["tool_calls"] = tool_calls
        if content:
            data["content"] = content
        if not data:
            data["content"] = "###STOP###"

        self.observer.observe("pi_bench_response_emitted", phase="respond", tool="a2a_artifact", state={"tool_calls": len(tool_calls)})
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=data))],
            name="openai_response",
        )
