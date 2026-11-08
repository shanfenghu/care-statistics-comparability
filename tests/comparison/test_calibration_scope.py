"""Numerical and policy checks for test calibration scope."""
from itertools import combinations
import math
import unittest

import numpy as np

from calibration_scope import scalar_answer, thresholds


class CalibrationScopeChecks(unittest.TestCase):
    def test_added_answers_can_be_wrong(self):
        # Another family's uncertainty widens both vector intervals. Removing
        # that requirement answers more but cannot repair a wrong point order.
        scores = np.tile([.01, .5], (39, 1))
        vector = thresholds(scores, .05, 'vector')
        family = thresholds(scores, .05, 'specified-family')
        pred, events = np.array([[100., 5.], [120., 5.]]), np.full((2, 2), 100.)
        self.assertEqual(scalar_answer(pred, events, vector, 0, 1, 0), 0)
        self.assertEqual(scalar_answer(pred, events, family, 0, 1, 0), -1)
        self.assertGreater(130 - 110, 0)  # Native order is opposite.

    def test_bonferroni_and_vector_have_no_universal_interval_order(self):
        scores = np.tile([0., 100.], (39, 1))
        self.assertLess(thresholds(scores, .1, 'Bonferroni-vector')[0],
                        thresholds(scores, .1, 'vector')[0])
        scores[:] = 0
        scores[-2:, 0] = 100
        self.assertGreater(thresholds(scores, .1, 'Bonferroni-vector')[0],
                           thresholds(scores, .1, 'vector')[0])

    def test_small_samples_keep_infinity(self):
        scores = np.ones((20, 2))
        self.assertTrue(np.isfinite(thresholds(scores, .05, 'vector')).all())
        self.assertTrue(np.isinf(thresholds(scores, .05, 'Bonferroni-vector')).all())
        self.assertTrue(np.isinf(thresholds(scores[:18], .05, 'specified-family')).all())

    def test_touching_intervals_abstain(self):
        pred = np.array([[100., 100.], [120., 120.]])
        events = np.full((2, 2), 100.)
        self.assertEqual(scalar_answer(pred, events, [.1, .1], 0, 1, 0), 0)
        self.assertEqual(scalar_answer(pred, events, [.05, .05], 0, 1, 0), -1)

    def test_bin_radius_prevents_false_zero_width_certainty(self):
        scores = np.full((39, 2), 2/100)  # Perfect centre fit, adult radius two.
        self.assertEqual(thresholds(scores, .05, 'specified-family').tolist(), [.02, .02])
        self.assertNotEqual(thresholds(scores, .05, 'specified-family')[0], 0)

    def test_family_permutation_preserves_policy(self):
        scores = np.array([[i, 40-i] for i in range(1, 40)], float)
        for scope in ('vector', 'specified-family', 'Bonferroni-vector'):
            np.testing.assert_array_equal(thresholds(scores[:, ::-1], .1, scope),
                                          thresholds(scores, .1, scope)[::-1])

    def test_rank_probability_from_all_label_positions(self):
        # Independent enumeration of assignments, not a simulation of quantiles.
        for n in range(1, 9):
            for m in (1, 2):
                for k in range(1, n+2):
                    success = total = 0
                    for test in combinations(range(n+m), m):
                        cal = [j for j in range(n+m) if j not in test]
                        cutoff = cal[k-1] if k <= n else math.inf
                        success += max(test) <= cutoff
                        total += 1
                    expected = math.comb(k+m-1, m)/math.comb(n+m, m)
                    self.assertAlmostEqual(success/total, expected)

    def test_ties_make_rank_coverage_conservative(self):
        values = [0, 0, 1, 1, 2]
        n, m, k = 3, 2, 2
        success = 0
        for test in combinations(range(5), m):
            cal = sorted(values[j] for j in range(5) if j not in test)
            success += all(values[j] <= cal[k-1] for j in test)
        self.assertGreaterEqual(success/10, k*(k+1)/((n+1)*(n+2)))


if __name__ == '__main__':
    unittest.main()
