"""Tests for dependency-free frame index selection."""

import unittest

from dlc.hybrid.frame_extraction import _evenly_spaced


class FrameExtractionTests(unittest.TestCase):
    def test_evenly_spaced_includes_both_ends(self) -> None:
        self.assertEqual(_evenly_spaced(list(range(10)), 3), [0, 4, 9])

    def test_does_not_duplicate_short_input(self) -> None:
        self.assertEqual(_evenly_spaced([2, 7], 5), [2, 7])


if __name__ == "__main__":
    unittest.main()
