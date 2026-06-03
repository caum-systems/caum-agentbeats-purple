from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Mapping, Optional


CONTINUE = "continue"
WATCH = "watch"
REPLAN = "replan"
BREAK_LOOP = "break_loop"


def _label(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text[:80] if text else default


@dataclass
class AdvisorDecision:
    action: str = CONTINUE
    tier: str = "T1"
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def should_hint(self) -> bool:
        return self.action in {WATCH, REPLAN, BREAK_LOOP}


class StructuralAdvisor:
    """
    Local zero-semantic structural advisor for benchmark assisted mode.

    It looks only at event/tool/status/state_id labels. It does not read task
    text, prompts, tool arguments, outputs, commands, files, or business data.
    The advisor never blocks; it only emits a suggested structural hint.
    """

    def __init__(self, *, window: int = 12, warmup_steps: int = 5):
        self.window = max(6, int(window))
        self.warmup_steps = max(3, int(warmup_steps))
        self.exact: Deque[tuple[str, str, str, str]] = deque(maxlen=self.window)
        self.pattern: Deque[tuple[str, str, str]] = deque(maxlen=self.window)
        self.state_history: Deque[int] = deque(maxlen=self.window + 1)
        self.seen_states: set[str] = set()
        self.step_index = 0
        self.last_decision = AdvisorDecision()

    def observe(self, event: Mapping[str, Any]) -> AdvisorDecision:
        event_label = _label(event.get("event"), "event")
        tool = _label(event.get("tool") or event.get("tool_family"), "agent_tool")
        status = _label(event.get("status"), "unknown")
        state = _label(event.get("state_id"), f"step_{self.step_index}")

        self.step_index += 1
        self.exact.append((event_label, tool, status, state))
        self.pattern.append((event_label, tool, status))
        self.seen_states.add(state)
        self.state_history.append(len(self.seen_states))
        self.last_decision = self._decide()
        return self.last_decision

    def _decide(self) -> AdvisorDecision:
        if self.step_index < self.warmup_steps:
            return AdvisorDecision(CONTINUE, "T1", 0.0, ["warmup_floor"])

        exact_counts = Counter(self.exact)
        pattern_counts = Counter(self.pattern)
        exact_repeat = max(exact_counts.values(), default=1)
        pattern_repeat = max(pattern_counts.values(), default=1)
        error_ratio = sum(1 for item in self.pattern if item[2] == "error") / max(len(self.pattern), 1)

        if exact_repeat >= 4:
            confidence = min(0.95, 0.55 + 0.10 * exact_repeat)
            return AdvisorDecision(
                BREAK_LOOP,
                "T4" if confidence >= 0.75 else "T3",
                round(confidence, 4),
                [f"exact_structural_repeat:{exact_repeat}x"],
            )

        if pattern_repeat >= 5 and error_ratio >= 0.40:
            confidence = min(0.80, 0.45 + 0.07 * pattern_repeat)
            return AdvisorDecision(
                REPLAN,
                "T3",
                round(confidence, 4),
                [f"error_pattern_repeat:{pattern_repeat}x", f"error_ratio:{error_ratio:.2f}"],
            )

        if len(self.state_history) >= self.window:
            growth = self.state_history[-1] - self.state_history[0]
            if growth <= 1:
                return AdvisorDecision(
                    WATCH,
                    "T2",
                    0.30,
                    [f"novelty_saturation:+{growth}_states_in_{len(self.state_history)-1}_steps"],
                )

        return AdvisorDecision(CONTINUE, "T1", 0.0, [])

    def hint_text(self) -> Optional[str]:
        decision = self.last_decision
        if not decision.should_hint:
            return None
        return (
            f"Structural review signal: {decision.action} ({decision.tier}, "
            f"confidence={decision.confidence:.2f}). Recent execution shows "
            "repetition, error-pattern pressure, or novelty saturation. Review "
            "whether to change strategy, shorten retries, or finalize directly."
        )
