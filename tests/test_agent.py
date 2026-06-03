from __future__ import annotations

import json
from types import SimpleNamespace

from caum_agentbeats_purple.action_format import parse_action
from caum_agentbeats_purple.agent import benchmark_context_messages, format_tool_calls


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
