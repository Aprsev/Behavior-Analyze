"""Stateful terminal-to-plain-text conversion for chunked QProcess output."""

from __future__ import annotations

import re

_DYNAMIC_PROGRESS = re.compile(r"\b\d{1,3}%\s+[━─╸╺]+")


def _keep_line(line: str) -> bool:
    """Hide transient progress bars while retaining final metric summaries."""

    return not _DYNAMIC_PROGRESS.search(line)


class TerminalStreamSanitizer:
    """Remove split ANSI sequences and emulate CR/CRLF terminal behavior."""

    def __init__(self) -> None:
        self.state = "text"
        self.line: list[str] = []
        self.pending_cr = False

    def _finish_line(self) -> str:
        value = "".join(self.line)
        self.line.clear()
        return value + "\n" if _keep_line(value) else ""

    def feed(self, text: str) -> str:
        output: list[str] = []
        for character in text:
            if self.state == "csi":
                if "@" <= character <= "~":
                    self.state = "text"
                continue
            if self.state == "osc":
                if character == "\x07":
                    self.state = "text"
                elif character == "\x1b":
                    self.state = "osc_escape"
                continue
            if self.state == "osc_escape":
                self.state = "text" if character == "\\" else "osc"
                continue
            if self.state == "escape":
                if character == "[":
                    self.state = "csi"
                elif character == "]":
                    self.state = "osc"
                else:
                    self.state = "text"
                continue

            if self.pending_cr:
                self.pending_cr = False
                if character == "\n":
                    output.append(self._finish_line())
                    continue
                # A bare CR means the terminal overwrites the current progress line.
                self.line.clear()

            if character == "\x1b":
                self.state = "escape"
            elif character == "\r":
                self.pending_cr = True
            elif character == "\n":
                output.append(self._finish_line())
            elif character == "\t" or ord(character) >= 32:
                self.line.append(character)
        return "".join(output)

    def flush(self) -> str:
        self.pending_cr = False
        if not self.line:
            return ""
        value = "".join(self.line)
        self.line.clear()
        return value if _keep_line(value) else ""
