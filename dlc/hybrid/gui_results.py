"""Parse structured result messages emitted by hybrid and legacy DLC jobs."""

from __future__ import annotations

import json

RESULT_PREFIXES = ("HYBRID_GUI_RESULT ", "DLC_GUI_RESULT ")


def parse_result_line(line: str) -> dict | None:
    for prefix in RESULT_PREFIXES:
        if line.startswith(prefix):
            value = json.loads(line.removeprefix(prefix))
            if not isinstance(value, dict):
                raise ValueError("GUI result payload must be a JSON object")
            return value
    return None
