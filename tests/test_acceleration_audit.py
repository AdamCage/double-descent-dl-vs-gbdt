import unittest
import numpy as np
from src.acceleration_audit import slope_scan, rank_threshold, binomial_interval, noisy_curve


class AccelerationAuditTests(unittest.TestCase):
    def test_affine_invariance_and_boundary(self):
        x = np.arange(51, dtype=float)
        y = 2 - .001 * x - .02 * np.maximum(x - 30, 0)
        result = slope_scan(y, [4, 8])
        self.assertEqual(result["t_star"], 30)
        self.assertAlmostEqual(result["score"], slope_scan(.01 * y + 10, [4, 8])["score"], places=10)

    def test_no_acceleration_for_affine_or_constant_loss(self):
        self.assertEqual(slope_scan(np.ones(21), [3])["score"], 0)
        self.assertLess(slope_scan(2 - .1 * np.arange(21), [3])["score"], 1e-14)

    def test_rank_bound_and_ties(self):
        result = rank_threshold(range(19), .1)
        self.assertEqual(result["order"], 18)
        self.assertEqual(result["threshold"], 17)
        self.assertEqual(rank_threshold([1] * 19, .1)["threshold"], 1)
        with self.assertRaises(ValueError):
            rank_threshold([1, 2], .01)

    def test_validation(self):
        for y, w in (([1, np.nan, 2, 3, 4], [2]), ([1] * 5, [3]), ([1] * 5, [True])):
            with self.assertRaises(ValueError):
                slope_scan(y, w)

    def test_noise_replay_and_independent_phase(self):
        base = np.zeros(20)
        a = noisy_curve(base, .1, .8, [99, 1])
        np.testing.assert_array_equal(a, noisy_curve(base, .1, .8, [99, 1]))
        self.assertFalse(np.array_equal(a, noisy_curve(base, .1, .8, [99, 2])))

    def test_binomial_edges(self):
        self.assertEqual(binomial_interval(0, 20)[0], 0)
        self.assertEqual(binomial_interval(20, 20)[1], 1)
        self.assertAlmostEqual(binomial_interval(0, 20)[1], 1 - .025 ** (1 / 20))
