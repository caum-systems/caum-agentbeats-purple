from __future__ import annotations

import json
from types import SimpleNamespace

from caum_agentbeats_purple.action_format import parse_action
from caum_agentbeats_purple.agent import (
    benchmark_context_messages,
    format_tool_calls,
    maybe_strip_tool_content,
    reorder_tool_calls,
    tool_schema_name,
    tool_strategy_message,
)


def test_parse_action_accepts_json_action():
    parsed = parse_action('{"name":"respond","arguments":{"content":"ok"}}')
    assert parsed["name"] == "respond"
    assert parsed["arguments"]["content"] == "ok"


def test_parse_action_wraps_plain_json():
    parsed = parse_action('{"content":"ok"}')
    assert parsed["name"] == "respond"
    assert parsed["arguments"]["content"] == "ok"


def test_parse_action_wraps_non_object_json():
    parsed = parse_action(json.dumps(["x"]))
    assert parsed["name"] == "respond"
    assert "x" in parsed["arguments"]["content"]


def test_benchmark_context_messages_prepends_policy_context():
    payload = {
        "benchmark_context": [
            {"kind": "policy", "content": "Use record_decision."},
            {"kind": "empty", "content": ""},
        ]
    }
    messages = benchmark_context_messages(payload)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "Pi-Bench policy agent" in messages[0]["content"]
    assert "### Policy" in messages[0]["content"]
    assert "Use record_decision." in messages[0]["content"]


def test_format_tool_calls_uses_pi_bench_flat_shape():
    call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="record_decision", arguments='{"decision":"DENY"}'),
    )
    formatted = format_tool_calls([call])
    assert formatted == [
        {
            "id": "call_1",
            "name": "record_decision",
            "arguments": '{"decision":"DENY"}',
        }
    ]


def test_tool_schema_name_reads_openai_function_schema():
    schema = {"type": "function", "function": {"name": "record_decision", "parameters": {}}}
    assert tool_schema_name(schema) == "record_decision"


def test_tool_strategy_message_adds_helpdesk_ordering_guidance():
    tools = [
        {"type": "function", "function": {"name": "record_decision"}},
        {"type": "function", "function": {"name": "unlock_account"}},
        {"type": "function", "function": {"name": "reset_password"}},
        {"type": "function", "function": {"name": "log_ticket"}},
    ]
    message = tool_strategy_message(tools)
    assert message is not None
    assert message["role"] == "system"
    assert "record_decision should be the last tool call" in message["content"]
    assert "unlock_account before reset_password" in message["content"]


def test_reorder_tool_calls_moves_record_decision_after_actions():
    calls = [
        {"id": "3", "name": "record_decision", "arguments": "{}"},
        {"id": "2", "name": "reset_password", "arguments": "{}"},
        {"id": "1", "name": "unlock_account", "arguments": "{}"},
        {"id": "4", "name": "log_ticket", "arguments": "{}"},
    ]
    reordered = reorder_tool_calls(calls)
    assert [call["name"] for call in reordered] == [
        "unlock_account",
        "reset_password",
        "log_ticket",
        "record_decision",
    ]


def test_reorder_tool_calls_applies_compliance_sequence():
    calls = [
        {"id": "4", "name": "record_decision", "arguments": "{}"},
        {"id": "3", "name": "escalate_to_compliance", "arguments": "{}"},
        {"id": "1", "name": "hold_transaction", "arguments": "{}"},
        {"id": "2", "name": "open_case", "arguments": "{}"},
    ]
    reordered = reorder_tool_calls(calls)
    assert [call["name"] for call in reordered] == [
        "hold_transaction",
        "open_case",
        "escalate_to_compliance",
        "record_decision",
    ]


def test_maybe_strip_tool_content_defaults_to_tool_only_turn(monkeypatch):
    monkeypatch.delenv("CAUM_AGENTBEATS_STRIP_TOOL_CONTENT", raising=False)
    content = maybe_strip_tool_content("I did the thing.", [{"name": "record_decision"}])
    assert content is None


def test_maybe_strip_tool_content_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CAUM_AGENTBEATS_STRIP_TOOL_CONTENT", "0")
    content = maybe_strip_tool_content("I did the thing.", [{"name": "record_decision"}])
    assert content == "I did the thing."
