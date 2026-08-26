import unittest

from tracking.head_fusion import choose_reflection


class ReflectionFusionTests(unittest.TestCase):
    def test_usable_bright_spot_rejects_disagreeing_model(self):
        result = choose_reflection((20, 20), .25, (80, 20), .90)
        self.assertEqual(result.point, (20.0, 20.0))
        self.assertEqual(result.source, "reflection_heuristic_rejects_model")

    def test_weak_bright_spot_does_not_override_model(self):
        result = choose_reflection((20, 20), .05, (80, 20), .90)
        self.assertEqual(result.point, (80.0, 20.0))
        self.assertEqual(result.source, "reflection_model_disagrees")


if __name__ == "__main__":
    unittest.main()
