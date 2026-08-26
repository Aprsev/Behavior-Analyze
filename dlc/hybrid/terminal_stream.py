"""Stateful terminal-to-plain-text conversion for the GUI live log."""

from __future__ import annotations

import re

_DYNAMIC_PROGRESS = re.compile(r"\b\d{1,3}%\s+[━─╸╺]+")
_TQDM_PROGRESS = re.compile(r"(?P<percent>\d{1,3})%.*?(?P<done>\d+)\s*/\s*(?P<total>\d+)")
_DLC_STAGES = {
    "Running detector with batch size": "DLC detector",
    "Running pose prediction with batch size": "DLC pose",
}


class TerminalStreamSanitizer:
    """Remove ANSI control codes and turn DLC tqdm updates into sparse log lines."""

    def __init__(self, progress_step: int = 5) -> None:
        self.state = "text"
        self.line: list[str] = []
        self.pending_cr = False
        self.progress_step = max(1, int(progress_step))
        self.dlc_stage: str | None = None
        self.last_progress = 0

    def _observe_stage(self, line: str) -> None:
        for marker, stage in _DLC_STAGES.items():
            if marker in line:
                self.dlc_stage = stage
                self.last_progress = 0
                return

    def _render_line(self, line: str, transient: bool = False) -> str:
        self._observe_stage(line)
        if self.dlc_stage:
            match = _TQDM_PROGRESS.search(line)
            if match:
                percent = min(100, int(match.group("percent")))
                bucket = 100 if percent == 100 else (percent // self.progress_step) * self.progress_step
                if bucket >= self.progress_step and bucket > self.last_progress:
                    self.last_progress = bucket
                    rendered = (
                        f"{self.dlc_stage} progress: {bucket}% "
                        f"({match.group('done')}/{match.group('total')})\n"
                    )
                    if percent == 100:
                        self.dlc_stage = None
                    return rendered
                return ""
        if _DYNAMIC_PROGRESS.search(line):
            return ""
        return "" if transient else line + "\n"

    def _finish_line(self, transient: bool = False) -> str:
        value = "".join(self.line)
        self.line.clear()
        return self._render_line(value, transient)

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
                output.append(self._finish_line(transient=True))

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
        if self.pending_cr:
            self.pending_cr = False
            return self._finish_line(transient=True)
        if not self.line:
            return ""
        rendered = self._finish_line()
        return rendered[:-1] if rendered.endswith("\n") else rendered
