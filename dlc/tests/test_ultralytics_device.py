"""Regression tests for portable Ultralytics device selection."""

from types import SimpleNamespace
import unittest

from dlc.hybrid.device import normalize_yolo_device, resolve_ultralytics_device


def fake_torch(*, cuda: bool = False, count: int = 0, mps: bool = False):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: count,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


class UltralyticsDeviceTests(unittest.TestCase):
    def test_auto_selects_first_cuda_gpu(self) -> None:
        device = resolve_ultralytics_device("auto", fake_torch(cuda=True, count=1))
        self.assertEqual(device, 0)

    def test_auto_falls_back_to_mps_then_cpu(self) -> None:
        self.assertEqual(resolve_ultralytics_device("auto", fake_torch(mps=True)), "mps")
        self.assertEqual(resolve_ultralytics_device("auto", fake_torch()), "cpu")

    def test_explicit_devices_are_preserved(self) -> None:
        self.assertEqual(resolve_ultralytics_device("1"), 1)
        self.assertEqual(resolve_ultralytics_device("0,1"), "0,1")
        self.assertEqual(resolve_ultralytics_device("cpu"), "cpu")

    def test_only_yolo_device_is_normalized(self) -> None:
        cfg = {"yolo": {"device": "auto"}, "model": {"device": "auto"}}
        normalize_yolo_device(cfg, fake_torch(cuda=True, count=1))
        self.assertEqual(cfg["yolo"]["device"], 0)
        self.assertEqual(cfg["model"]["device"], "auto")


if __name__ == "__main__":
    unittest.main()
