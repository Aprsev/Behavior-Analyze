"""DLC progress compression in the GUI live log."""

import unittest

from dlc.hybrid.terminal_stream import TerminalStreamSanitizer


class DlcProgressLogTests(unittest.TestCase):
    def test_detector_progress_is_reported_every_five_percent(self) -> None:
        stream = TerminalStreamSanitizer()
        output = stream.feed(
            "Running detector with batch size 1\n"
            "  1% ━ 82/8164\r  4% ━ 327/8164\r"
            "  5% ━ 409/8164\r  9% ━ 735/8164\r"
            " 10% ━ 816/8164\r100% ━ 8164/8164\r\n"
        )
        self.assertIn("DLC detector progress: 5% (409/8164)\n", output)
        self.assertIn("DLC detector progress: 10% (816/8164)\n", output)
        self.assertIn("DLC detector progress: 100% (8164/8164)\n", output)
        self.assertNotIn("progress: 1%", output)


if __name__ == "__main__":
    unittest.main()
