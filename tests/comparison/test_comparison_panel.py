"""Numerical and policy checks for test comparison panel."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np

from comparison_panel import case_results, conformal_bounds, conformal_quantile, features, fit_predict, order
from comparison_robustness import queries, run_case, top_budget
from calibration_scope import evaluate


class ComparisonTests(unittest.TestCase):
    def row(self, i):
        return dict(code=str(i), year=2025, region="region-a", e=[100+i*10, 200+i*7],
                    p=[75+i*6, 150+i*4], radius=2,
                    previous=dict(year=2024, e=[95+i*10, 195+i*7], p=[70+i*6, 145+i*4]))

    def test_budget_respects_floor_and_stable_ties(self):
        answer = top_budget(np.array([1, -1, 1, -1, 1]), np.ones(5), .5)
        np.testing.assert_array_equal(answer, [1, -1, 0, 0, 0])
        # A selected point tie is still an abstention, never a fabricated sign.
        np.testing.assert_array_equal(top_budget(np.array([0, 1]), np.array([2., 1.]), .5), [0, 0])

    def test_invalid_budget_fractions_are_rejected(self):
        point = np.array([1, -1, 1, -1])
        score = np.array([4., 3., 2., 1.])
        for budget in (-.5, 1.5, math.nan, math.inf):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                top_budget(point, score, budget)
        np.testing.assert_array_equal(top_budget(point, score, 0), np.zeros(4))
        np.testing.assert_array_equal(top_budget(point, score, 1), point)

    def test_budget_margins_do_not_read_current_native_targets(self):
        a = [self.row(i) for i in range(4)]
        b = deepcopy(a)
        for row in b:row['p'] = [999999, 1]
        pred = np.array([[71, 150], [91, 140], [95, 180], [97, 190]], dtype=float)
        for temporal in (False, True):
            for q, r in zip(queries(a, pred, temporal), queries(b, pred, temporal)):
                np.testing.assert_array_equal(q['score'], r['score'])
                np.testing.assert_array_equal(top_budget(q['point'], q['score'], .5),
                                              top_budget(r['point'], r['score'], .5))

    def test_rank_and_small_calibration_set(self):
        self.assertEqual(conformal_quantile([1, 2, 3, 4, 5], .2), 5)
        self.assertTrue(math.isinf(conformal_quantile([1, 2, 3, 4, 5], .05)))

    def test_unknown_count_estimator_and_wrong_task_are_rejected(self):
        train = [self.row(i) for i in range(25)]
        evaluate = [self.row(27)]
        cfg = dict(ridge_alpha=1, seed=20261103,
                   forest=dict(n_estimators=2, min_samples_leaf=3, n_jobs=1))
        with self.assertRaisesRegex(ValueError, 'Unknown count estimator'):
            fit_predict('context_forets', train, evaluate, 0, cfg, False)
        with self.assertRaisesRegex(ValueError, 'temporal task'):
            fit_predict('lagged_factor', train, evaluate, 0, cfg, False)

    def test_invalid_calibration_levels_and_scores_are_rejected(self):
        for alpha in (-.1, 0, 1, 2, math.nan, math.inf, True, False):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                conformal_quantile([1, 2, 3], alpha)
        for scores in ([-1, 2], [math.nan], [math.inf], [[1, 2]], 1.):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                conformal_quantile(scores, .05)
        self.assertTrue(math.isinf(conformal_quantile([], .05)))

    def test_group_calibration_covers_both_native_bin_endpoints(self):
        rows = [self.row(i) for i in range(20)]
        prediction = np.array([r["p"] for r in rows], dtype=float)
        prediction[0, 1] += 20
        q = conformal_bounds(prediction, rows, .05)
        self.assertAlmostEqual(q, 22/200)
        e = np.array([r["e"] for r in rows])
        p = np.array([r["p"] for r in rows])
        self.assertTrue(np.all(prediction-q*e <= p-2))
        self.assertTrue(np.all(prediction+q*e >= p+2))

    def test_touching_intervals_abstain_and_reversing_endpoints_changes_sign(self):
        self.assertEqual(int(order(10, 12, 12, 14)), 0)
        self.assertEqual(int(order(10, 12, 13, 14)), -1)
        self.assertEqual(int(order(13, 14, 10, 12)), 1)

    def test_zero_truncated_native_bin_can_make_score_conservative(self):
        # At P=1, r=2 the native bin is [0,3], not [-1,3]. A prediction of
        # 10 has maximum native-bin discrepancy 10, while the score uses 11.
        rows = [dict(e=[10, 10], p=[1, 1], radius=2) for _ in range(20)]
        prediction = np.full((20, 2), 10.)
        q = conformal_bounds(prediction, rows, .05)
        native_low, native_high = 0., 3.
        exact_radius = max(abs(10.-native_low), abs(10.-native_high))
        self.assertAlmostEqual(q, 1.1)
        self.assertGreater(q*10, exact_radius)
        self.assertLessEqual(max(0., 10.-q*10), native_low)
        self.assertGreaterEqual(10.+q*10, native_high)

    def zero_truncated_case(self):
        """Use fixed predictions whose finite intervals are exactly [0, 3]."""
        rows = [dict(source='boundary', code=str(i), year=2025, region='a',
                     e=[10, 10], p=[1, 1], radius=2) for i in range(44)]
        return dict(name='boundary', source='boundary', task='geographic',
                    group='boundary', train=rows[:2], cal=rows[2:42], test=rows[42:])

    def zero_truncated_replay(self):
        case = self.zero_truncated_case()
        with patch('comparison_robustness.fit_predict',
                   side_effect=lambda method, train, rows, family, cfg, temporal:
                   np.ones(len(rows))):
            return run_case(case, dict(models=['literal']),
                            dict(alpha_values=[.025, .05, .1], answer_budgets=[]))

    def test_panel_coverage_uses_zero_truncated_native_bin(self):
        rows = self.zero_truncated_case()['test']
        pred = np.ones((2, 2))
        covered, _ = case_results('boundary', 'literal', rows, pred, .2, False)
        self.assertTrue(all(r['covered'] and r['group_covered'] for r in covered))
        # Clipping the lower endpoint must not conceal a missed upper endpoint.
        missed, _ = case_results('boundary', 'literal', rows, pred, .1, False)
        self.assertTrue(all(not r['covered'] for r in missed))

    def test_robustness_coverage_uses_zero_truncated_native_bin(self):
        result = self.zero_truncated_replay()
        for metric in result['methods']['literal']['metrics']:
            if metric['policy'].startswith('alpha='):
                with self.subTest(policy=metric['policy'], family=metric['family']):
                    self.assertEqual(metric['covered_rows'], 2)
                    self.assertEqual(metric['covered_vectors'], 2)

    def test_scope_coverage_uses_zero_truncated_native_bin(self):
        records, _ = evaluate(self.zero_truncated_replay())
        for record in records:
            with self.subTest(alpha=record['alpha'], scope=record['scope'],
                              family=record['family']):
                self.assertEqual(record['covered_rows'], 2)
                self.assertEqual(record['covered_vectors'], 2)

    def test_structural_count_bound_can_certify_temporal_decline(self):
        self.assertEqual(int(order(0, 5, 10, 10)), -1)
        self.assertEqual(int(order(0, 15, 10, 10)), 0)

    def test_covered_endpoints_imply_correct_guarded_pair(self):
        # Exhaustive small integers: a returned order is correct whenever both
        # true values belong to their prediction intervals, without independence.
        for alo in range(4):
            for ahi in range(alo, 4):
                for blo in range(4):
                    for bhi in range(blo, 4):
                        direction = int(order(alo, ahi, blo, bhi))
                        if direction:
                            for a in range(alo, ahi+1):
                                for b in range(blo, bhi+1):
                                    self.assertEqual(direction, int(np.sign(a-b)))

    def test_current_reference_values_are_not_features(self):
        a = self.row(2); b = deepcopy(a); b["p"] = [1, 999999]
        self.assertEqual(features(a, 0), features(b, 0))
        self.assertEqual(features(a, 0, True), features(b, 0, True))
        b["previous"]["p"][0] += 10
        self.assertNotEqual(features(a, 0, True), features(b, 0, True))

    def test_predictions_do_not_read_evaluation_targets(self):
        train = [self.row(i) for i in range(25)]
        a = [self.row(27)]; b = deepcopy(a); b[0]["p"] = [1000000, 1]
        cfg = dict(ridge_alpha=1, seed=20261103,
                   forest=dict(n_estimators=5, min_samples_leaf=3, n_jobs=1))
        for temporal in (False, True):
            methods = ["literal", "family_factor", "isotonic", "context_ridge", "context_forest"]
            if temporal:methods.append("lagged_factor")
            for method in methods:
                np.testing.assert_array_equal(fit_predict(method, train, a, 0, cfg, temporal),
                                              fit_predict(method, train, b, 0, cfg, temporal))


if __name__ == "__main__":
    unittest.main()
