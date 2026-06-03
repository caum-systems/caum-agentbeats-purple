from __future__ import annotations

import json

from caum_agentbeats_purple.caum_layer import CaumStructuralObserver, stable_hash, summarize_receipt


def test_stable_hash_is_order_invariant():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_local_observer_writes_zero_semantic_trace(tmp_path):
    observer = CaumStructuralObserver(mode="baseline", trace_dir=tmp_path)
    observer.observe("task_received", tool="a2a_message", state={"task": "private text already hashed by caller"})

    rows = list(observer.trace.path.read_text(encoding="utf-8").splitlines())
    assert len(rows) == 1
    record = json.loads(rows[0])
    event = record["event"]
    assert event["event"] == "task_received"
    assert event["tool"] == "a2a_message"
    assert event["client_sanitized"] if "client_sanitized" in event else True
    assert "private text" not in rows[0]


def test_structural_hint_only_in_assisted_pressure():
    observer = CaumStructuralObserver(mode="assisted")
    observer.last_receipt = {"tier": {"tier": "T4"}, "live_alert": False, "public_class": "review_only"}
    hint = observer.hint()
    assert hint.should_review_strategy
    assert hint.hint


def test_control_mode_emits_stronger_hint():
    observer = CaumStructuralObserver(mode="control")
    observer.last_receipt = {"tier": {"tier": "T4"}, "live_alert": False, "public_class": "review_only"}
    hint = observer.hint()
    assert hint.should_review_strategy
    assert "CAUM Control signal" in (hint.hint or "")
    assert "Switch strategy" in (hint.hint or "")


def test_summarize_receipt_is_small():
    summary = summarize_receipt(
        {
            "tier": {"tier": "T5"},
            "live_alert": True,
            "public_class": "hard_alert",
            "receipt_id": "r1",
            "raw_content": "must_not_copy",
        }
    )
    assert summary == {
        "tier": "T5",
        "live_alert": True,
        "public_class": "hard_alert",
        "receipt_id": "r1",
        "event_type": None,
    }
