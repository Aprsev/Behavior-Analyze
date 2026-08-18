#!/usr/bin/env python3
"""Checkpoint output compatibility tests for U-Net v1/v2/v3."""
import unittest

try:
    import torch
    from model import UNet, checkpoint_model, unpack_outputs
except ImportError:  # local CPU workstation may intentionally lack PyTorch
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelCompatibilityTests(unittest.TestCase):
    def package(self, model, **metadata):
        return {"state_dict": model.state_dict(), "size": 32,
                "in_channels": 1, **metadata}

    def test_v1_mask_only_checkpoint(self):
        model = checkpoint_model(self.package(UNet()), "cpu")
        mask, head, reflection = unpack_outputs(model(torch.zeros(1, 1, 32, 32)))
        self.assertEqual(tuple(mask.shape), (1, 1, 32, 32))
        self.assertIsNone(head); self.assertIsNone(reflection)

    def test_v2_head_checkpoint(self):
        source = UNet(head_output=True)
        model = checkpoint_model(self.package(source, head_output=True), "cpu")
        _, head, reflection = unpack_outputs(model(torch.zeros(1, 1, 32, 32)))
        self.assertIsNotNone(head); self.assertIsNone(reflection)

    def test_v3_reflection_checkpoint(self):
        source = UNet(head_output=True, reflection_output=True)
        model = checkpoint_model(self.package(
            source, head_output=True, reflection_output=True), "cpu")
        mask, head, reflection = unpack_outputs(model(torch.zeros(1, 1, 32, 32)))
        self.assertEqual(tuple(mask.shape), (1, 1, 32, 32))
        self.assertIsNotNone(head); self.assertIsNotNone(reflection)


if __name__ == "__main__":
    unittest.main()
