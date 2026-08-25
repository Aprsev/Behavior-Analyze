"""Sanitize subprocess output before it reaches the Qt live log."""

from __future__ import annotations

import re

# CSI (colors/progress), OSC (terminal titles/links), then other two-byte ESC.
ANSI = re.compile(
    r"\x1B(?:\][^\x07]*(?:\x07|\x1B\\)|\[[0-?]*[ -/]*[@-~]|[@-_])"
)


def sanitize_process_output(text: str) -> str:
    text = ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    return "".join(character for character in text if character in "\n\t" or ord(character) >= 32)
