from __future__ import annotations

import json

from caum_agentbeats_purple.action_format import parse_action


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
