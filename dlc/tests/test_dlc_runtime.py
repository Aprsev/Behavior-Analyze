"""Tests for hybrid DLC runtime safeguards."""

import unittest

from dlc.hybrid.dlc_runtime import validate_video_adaptation


class DlcRuntimeTests(unittest.TestCase):
    def test_rejects_zero_epoch_video_adaptation(self) -> None:
        with self.assertRaisesRegex(ValueError, "both adaptation epoch values"):
            validate_video_adaptation(
                {"model": {"video_adapt": True, "detector_epochs": 0, "pose_epochs": 0}}
            )

    def test_allows_plain_model_zoo_inference_with_zero_epochs(self) -> None:
        validate_video_adaptation(
            {"model": {"video_adapt": False, "detector_epochs": 0, "pose_epochs": 0}}
        )


if __name__ == "__main__":
    unittest.main()
