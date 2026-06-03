from __future__ import annotations

import json
from typing import Any


def parse_action(content: str) -> dict[str, Any]:
    parsed = json.loads(content or "{}")
    if not isinstance(parsed, dict):
        return {"name": "respond", "arguments": {"content": str(parsed)}}
    if "name" not in parsed:
        parsed = {"name": "respond", "arguments": parsed}
    if "arguments" not in parsed or not isinstance(parsed.get("arguments"), dict):
        parsed["arguments"] = {"content": str(parsed.get("arguments", ""))}
    return parsed
