from __future__ import annotations

import os
import time

from dotenv import load_dotenv
import litellm

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
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


class Agent:
    def __init__(self, observer: CaumStructuralObserver | None = None):
        self.model = os.getenv("CAUM_AGENT_LLM", os.getenv("TAU2_AGENT_LLM", "openai/gpt-4.1"))
        self.observer = observer or CaumStructuralObserver()
        self.messages: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
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
