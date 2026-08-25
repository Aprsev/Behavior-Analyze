"""Chunk-safe terminal stream cleanup for the GUI live log."""

import unittest

from dlc.hybrid.terminal_stream import TerminalStreamSanitizer


class TerminalStreamTests(unittest.TestCase):
    def test_split_ansi_sequence_is_removed(self) -> None:
        stream = TerminalStreamSanitizer()
        self.assertEqual(stream.feed("before\n\x1b[3"), "before\n")
        self.assertEqual(stream.feed("4mblue\x1b[0"), "")
        self.assertEqual(stream.feed("m\nafter\n"), "blue\nafter\n")

    def test_crlf_is_preserved_but_bare_cr_overwrites(self) -> None:
        stream = TerminalStreamSanitizer()
        self.assertEqual(stream.feed("normal\r"), "")
        self.assertEqual(stream.feed("\nold progress\rnew value\n"), "normal\nnew value\n")

    def test_dynamic_progress_bar_is_dropped(self) -> None:
        stream = TerminalStreamSanitizer()
        text = stream.feed("Class Images: 40% ━━━╸──── 2/5\nall 80 80 0.794 0.87\n")
        self.assertEqual(text, "all 80 80 0.794 0.87\n")


if __name__ == "__main__":
    unittest.main()
