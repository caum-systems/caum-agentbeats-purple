from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from .structural_advisor import StructuralAdvisor


DEFAULT_LIVE_URL = "https://caum-observation-production.up.railway.app/v2/live"
STRUCTURAL_PRESSURE_TIERS = {"T4", "T5"}


def stable_hash(value: Any, prefix: str = "h") -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


def safe_label(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    chars = []
    for ch in text[:80]:
        chars.append(ch if ch.isalnum() or ch in {"_", "-", ".", ":"} else "_")
    return "".join(chars).strip("_") or default


@dataclass
class StructuralHint:
    mode: str
    tier: Optional[str] = None
    live_alert: bool = False
    public_class: Optional[str] = None
    hint: Optional[str] = None

    @property
    def should_review_strategy(self) -> bool:
        return bool(self.live_alert or (self.tier or "").upper() in STRUCTURAL_PRESSURE_TIERS)


class LocalTraceWriter:
    def __init__(self, trace_dir: str | Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"caum_agentbeats_trace_{int(time.time())}_{uuid.uuid4().hex[:8]}.jsonl"

    def write(self, event: dict[str, Any], receipt: Optional[dict[str, Any]] = None) -> None:
        record = {
            "ts": time.time(),
            "event": event,
            "receipt_summary": summarize_receipt(receipt),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class DirectCaumLiveSession:
    """Small CAUM Live client used inside the Docker image."""

    def __init__(self, *, api_key: str, base_url: str, workflow: str, agent_id: str, task_family: str):
        self.base_url = base_url.rstrip("/")
        self.session_id = f"caum-agentbeats-{uuid.uuid4().hex[:16]}"
        self.session_token: Optional[str] = None
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self.workflow = workflow
        self.agent_id = agent_id
        self.task_family = task_family
        self.start()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}{path}", json=payload, headers=self.headers, timeout=20)
        if response.status_code >= 400:
            raise RuntimeError(f"CAUM Live API error {response.status_code}: {response.text[:200]}")
        return response.json()

    def start(self) -> None:
        result = self._post(
            "/start",
            {
                "session_id": self.session_id,
                "task_id": self.session_id,
                "agent_id": self.agent_id,
                "workflow": self.workflow,
                "task_family": self.task_family,
                "source": "agentbeats_purple_direct_client",
            },
        )
        self.session_id = result.get("session", {}).get("session_id") or self.session_id
        self.session_token = result.get("session_token")

    def event(self, event: dict[str, Any]) -> dict[str, Any]:
        if not self.session_token:
            self.start()
        return self._post(
            "/event",
            {
                "session_id": self.session_id,
                "session_token": self.session_token,
                "event": event,
            },
        )


def summarize_receipt(receipt: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        return {}
    tier_obj = receipt.get("tier") or {}
    tier = tier_obj.get("tier") if isinstance(tier_obj, dict) else receipt.get("current_tier")
    return {
        "tier": tier or receipt.get("current_tier"),
        "live_alert": bool(receipt.get("live_alert")),
        "public_class": receipt.get("public_class"),
        "receipt_id": receipt.get("receipt_id"),
        "event_type": receipt.get("event_type"),
    }


class CaumStructuralObserver:
    """
    Observe AgentBeats Purple Agent execution with CAUM.

    This class is intentionally zero-semantic. Callers pass only structural
    labels, counters, hashed state ids, and generic phase/status values.
    """

    def __init__(
        self,
        *,
        mode: Optional[str] = None,
        workflow: str = "caum_agentbeats_purple",
        agent_id: str = "caum_agentbeats_purple_v0_1",
        task_family: str = "agentbeats_purple_agent",
        trace_dir: Optional[str | Path] = None,
    ):
        self.mode = safe_label(mode or os.getenv("CAUM_AGENTBEATS_MODE", "baseline"), "baseline")
        self.workflow = workflow
        self.agent_id = agent_id
        self.task_family = task_family
        self.trace = LocalTraceWriter(trace_dir or os.getenv("CAUM_STRUCTURAL_TRACE_DIR", "./runs"))
        self.session: Any = None
        self.last_receipt: Optional[dict[str, Any]] = None
        self.advisor = StructuralAdvisor()
        self.last_advisor_decision: Any = None
        self.enabled_live = self.mode in {"observe", "assisted", "control"} and bool(os.getenv("CAUM_LIVE_API_KEY"))
        self._start_live_if_available()

    def _start_live_if_available(self) -> None:
        if not self.enabled_live:
            return
        try:
            self.session = DirectCaumLiveSession(
                api_key=os.getenv("CAUM_LIVE_API_KEY", ""),
                base_url=os.getenv("CAUM_LIVE_URL", DEFAULT_LIVE_URL),
                agent_id=self.agent_id,
                workflow=self.workflow,
                task_family=self.task_family,
            )
        except Exception as exc:
            self.enabled_live = False
            self.session = None
            self.trace.write(
                {
                    "event": "caum_live_start_error",
                    "tool": "caum_live",
                    "phase": "observe",
                    "status": "error",
                    "state_id": stable_hash(type(exc).__name__, "state"),
                    "lane": "agentbeats",
                    "tool_family": "agent_tool",
                },
                {"event_type": "caum_submission_error", "public_class": "review_only"},
            )

    def observe(
        self,
        event: str,
        *,
        tool: str = "agent",
        phase: str = "execute",
        status: str | bool = "ok",
        state: Optional[Any] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None,
        cost_usd: Optional[float] = None,
        lane: str = "agentbeats",
    ) -> Optional[dict[str, Any]]:
        structural = {
            "event": safe_label(event, "event"),
            "tool": safe_label(tool, "agent_tool"),
            "phase": safe_label(phase, "execute"),
            "status": status,
            "state_id": stable_hash(state if state is not None else {"event": event, "phase": phase}, "state"),
            "lane": safe_label(lane, "agentbeats"),
            "tool_family": "agent_tool",
        }
        if tokens_used is not None:
            structural["tokens_used"] = int(max(0, tokens_used))
        if latency_ms is not None:
            structural["latency_ms"] = int(max(0, latency_ms))
        if cost_usd is not None:
            structural["cost_usd"] = round(float(max(0.0, cost_usd)), 8)

        receipt: Optional[dict[str, Any]] = None
        if self.session is not None:
            try:
                receipt = self.session.event(structural)
                self.last_receipt = receipt
            except Exception as exc:
                receipt = {"event_type": "caum_submission_error", "public_class": "review_only", "error_hash": stable_hash(str(exc), "err")}
        self.last_advisor_decision = self.advisor.observe(structural)
        self.trace.write(structural, receipt)
        return receipt

    def hint(self) -> StructuralHint:
        summary = summarize_receipt(self.last_receipt)
        tier = summary.get("tier")
        hint = None
        if self.mode in {"assisted", "control"} and (summary.get("live_alert") or str(tier).upper() in STRUCTURAL_PRESSURE_TIERS):
            if self.mode == "control":
                hint = (
                    "CAUM Control signal: recent execution shows repetition or stall pressure. "
                    "Do not repeat the same approach. Switch strategy now, shorten the retry path, "
                    "or finalize if enough evidence exists."
                )
            else:
                hint = (
                    "Structural review signal: recent execution shows repetition or stall pressure. "
                    "Review whether a different strategy, shorter retry path, or direct finalization is appropriate."
                )
        if self.mode in {"assisted", "control"} and not hint:
            hint = self.advisor.hint_text()
            if hint and self.last_advisor_decision is not None:
                tier = getattr(self.last_advisor_decision, "tier", tier)
                if self.mode == "control":
                    hint = hint.replace("Review whether to", "Switch strategy now:").replace(
                        "Structural review signal", "CAUM Control signal"
                    )
        return StructuralHint(
            mode=self.mode,
            tier=tier,
            live_alert=bool(summary.get("live_alert")),
            public_class=summary.get("public_class"),
            hint=hint,
        )
