import unittest

import torch

from adaptive_test import execute_strategy


class FlattenModel:
    def __call__(self, frame):
        return frame.flatten(start_dim=1)


class AdaptiveCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = FlattenModel()
        values = [0.0, 0.01, 0.03, 0.031, 0.06]
        self.frames = [
            torch.full((1, 1, 1, 1), value, dtype=torch.float32)
            for value in values
        ]
        with torch.inference_mode():
            self.references = [self.model(frame) for frame in self.frames]

    def test_always_refresh_recomputes_every_frame(self):
        result = execute_strategy(
            "Always Refresh", self.model, self.frames, self.references
        )
        self.assertEqual(result["Refreshes"], len(self.frames))
        self.assertEqual(result["Cache_Hits"], 0)
        self.assertEqual(result["Hit_Rate"], 0.0)
        self.assertAlmostEqual(result["Mean_Feature_Error"], 0.0)

    def test_fixed_rate_refreshes_at_expected_indices(self):
        result = execute_strategy(
            "Fixed Rate",
            self.model,
            self.frames,
            self.references,
            refresh_interval=2,
        )
        self.assertEqual(result["Refreshes"], 3)  # indices 0, 2, and 4
        self.assertEqual(result["Cache_Hits"], 2)
        self.assertAlmostEqual(result["Hit_Rate"], 0.4)

    def test_adaptive_cache_compares_against_last_refreshed_frame(self):
        result = execute_strategy(
            "Adaptive Cache",
            self.model,
            self.frames,
            self.references,
            threshold=0.02,
        )
        # Refresh at values 0.00, 0.03, and 0.06.
        self.assertEqual(result["Refreshes"], 3)
        self.assertEqual(result["Cache_Hits"], 2)
        self.assertGreater(result["Mean_Feature_Error"], 0.0)

    def test_unknown_strategy_raises_error(self):
        with self.assertRaises(ValueError):
            execute_strategy(
                "Not a strategy", self.model, self.frames, self.references
            )


if __name__ == "__main__":
    unittest.main()
