from __future__ import annotations

from caum_agentbeats_purple.structural_advisor import BREAK_LOOP, REPLAN, WATCH, StructuralAdvisor


def test_advisor_breaks_exact_repeat_after_warmup():
    advisor = StructuralAdvisor(warmup_steps=3)
    decision = None
    for _ in range(6):
        decision = advisor.observe({"event": "tool_result", "tool": "shell", "status": "error", "state_id": "same"})
    assert decision is not None
    assert decision.action == BREAK_LOOP
    assert decision.tier in {"T3", "T4"}


def test_advisor_replans_repeated_error_pattern_with_state_jitter():
    advisor = StructuralAdvisor(warmup_steps=3)
    decision = None
    for i in range(7):
        decision = advisor.observe({"event": "tool_result", "tool": "shell", "status": "error", "state_id": f"s{i}"})
    assert decision is not None
    assert decision.action == REPLAN
    assert "error_pattern_repeat" in " ".join(decision.reasons)


def test_advisor_watch_on_novelty_saturation():
    advisor = StructuralAdvisor(window=6, warmup_steps=3)
    decision = None
    for i in range(8):
        decision = advisor.observe({"event": "observation", "tool": "agent", "status": "ok", "state_id": f"s{i % 2}"})
    assert decision is not None
    assert decision.action in {WATCH, BREAK_LOOP}


def test_advisor_hint_text_is_structural_only():
    advisor = StructuralAdvisor(warmup_steps=3)
    for _ in range(6):
        advisor.observe({"event": "tool_result", "tool": "shell", "status": "error", "state_id": "same"})
    hint = advisor.hint_text()
    assert hint
    assert "Structural review signal" in hint
