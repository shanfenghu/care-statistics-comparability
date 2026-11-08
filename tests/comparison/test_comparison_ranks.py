"""Numerical and policy checks for test comparison ranks."""
from fractions import Fraction
from itertools import combinations
import unittest

from verify_comparison_ranks import rank_bounds


class RankBounds(unittest.TestCase):
    def test_distinct_score_orbits_attain_pair_coefficient(self):
        for n in range(1, 9):
            for k in range(1, n+2):
                alpha = 1-Fraction(2*k-1, 2*(n+1))
                actual_k, single, pair = rank_bounds(n, alpha)
                self.assertEqual(actual_k, k)
                # Every labelled pair of rank positions has equal mass.
                total = failed = first_failed = 0
                for a, b in combinations(range(n+2), 2):
                    calibration = [s for s in range(n+2) if s not in (a, b)]
                    q = calibration[k-1] if k <= n else float('inf')
                    total += 2
                    failed += 2*int(max(a, b) > q)
                    first_failed += int(a > q)+int(b > q)
                self.assertEqual(Fraction(failed, total), pair)
                self.assertEqual(Fraction(first_failed, total), single)
                self.assertLessEqual(pair, 2*single)
                self.assertLessEqual(2*single, 2*alpha)

    def test_ties_are_conservative_on_labelled_position_orbits(self):
        for scores in ([0, 0, 1], [0, 0, 1, 1, 2], [1]*6):
            n = len(scores)-2
            for k in range(1, n+2):
                alpha = 1-Fraction(2*k-1, 2*(n+1))
                _, _, pair = rank_bounds(n, alpha)
                total = failed = 0
                for a, b in combinations(range(n+2), 2):
                    calibration = sorted(s for i, s in enumerate(scores) if i not in (a, b))
                    q = calibration[k-1] if k <= n else float('inf')
                    total += 1; failed += max(scores[a], scores[b]) > q
                self.assertLessEqual(Fraction(failed, total), pair)

    def test_infinite_rank_boundary_and_known_small_orbit(self):
        for n in range(1, 40):
            k, single, pair = rank_bounds(n, Fraction(1, 2*(n+1)))
            self.assertEqual((k, single, pair), (n+1, 0, 0))
            self.assertEqual(rank_bounds(n, Fraction(1, n+1))[0], n)
        self.assertEqual(rank_bounds(3, Fraction(1, 4)), (3, Fraction(1, 4), Fraction(2, 5)))
        self.assertEqual(rank_bounds(30, Fraction(1, 20))[2], Fraction(1, 16))

    def test_separate_marginal_exchangeability_does_not_imply_joint_bound(self):
        # C,X are independent uniforms on ten scores; Y=9-X. Each of (C,X)
        # and (C,Y) is exchangeable, but (C,X,Y) is not. With n=k=1, q=C.
        tuples = [(c, x, 9-x) for c in range(10) for x in range(10)]
        first = sum(x <= c for c, x, y in tuples)
        second = sum(y <= c for c, x, y in tuples)
        joint = sum(x <= c and y <= c for c, x, y in tuples)
        _, single, pair = rank_bounds(1, Fraction(1, 2))
        self.assertGreaterEqual(Fraction(first, len(tuples)), 1-single)
        self.assertGreaterEqual(Fraction(second, len(tuples)), 1-single)
        self.assertEqual(Fraction(joint, len(tuples)), Fraction(3, 10))
        self.assertLess(Fraction(joint, len(tuples)), 1-pair)

    def test_invalid_domains_are_rejected(self):
        for n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                rank_bounds(n, Fraction(1, 20))
        for alpha in (Fraction(0), Fraction(1), Fraction(-1), .05):
            with self.assertRaises(ValueError):
                rank_bounds(20, alpha)


if __name__ == '__main__':
    unittest.main()
