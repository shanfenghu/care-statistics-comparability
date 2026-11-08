"""Independent witness and exhaustive small-world checks of structural logic."""
import itertools
from fractions import Fraction
import unittest

from countviews import Obstruction, Rewrite, derive
from countviews.values import parse_count


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class LinearTests(unittest.TestCase):
    def verify(self, a, q):
        result = derive(a, q)
        if isinstance(result, Rewrite):
            for j in range(len(q)):
                self.assertEqual(sum(w * row[j] for w, row in zip(result.coefficients, a)), q[j])
            # Compare with direct evaluation on all small databases, not another solver.
            for x in itertools.product(range(3), repeat=len(q)):
                self.assertEqual(result.evaluate([dot(row, x) for row in a]), dot(q, x))
        else:
            self.assertTrue(all(isinstance(v, int) and v >= 0 for v in result.left + result.right))
            self.assertEqual([dot(r, result.left) for r in a], [dot(r, result.right) for r in a])
            self.assertNotEqual(dot(q, result.left), dot(q, result.right))
        return result

    def test_all_binary_three_atom_view_families(self):
        rows = list(itertools.product(range(2), repeat=3))
        for a in itertools.product(rows, repeat=2):
            for q in rows:
                self.verify(a, q)

    def test_rational_rewrite_and_duplicates(self):
        r = self.verify([[2, 0], [2, 0], [0, 3]], [1, -1])
        self.assertEqual(r.coefficients, (Fraction(1, 2), 0, Fraction(-1, 3)))

    def test_empty_views_zero_query_and_missing_query(self):
        self.assertIsInstance(self.verify([], [0, 0]), Rewrite)
        self.assertIsInstance(self.verify([], [1, 0]), Obstruction)

    def test_union_needs_overlap(self):
        self.assertIsInstance(self.verify([[1, 1, 0], [0, 1, 1]], [1, 1, 1]), Obstruction)
        self.assertIsInstance(self.verify([[1, 1, 0], [0, 1, 1], [0, 1, 0]], [1, 1, 1]), Rewrite)

    def test_malformed_shapes_and_types(self):
        for a, q in [([[1]], [1, 0]), ([], [])]:
            with self.assertRaises(ValueError):
                derive(a, q)
        with self.assertRaises(TypeError):
            derive([[1.0]], [1])
        with self.assertRaises(ValueError):
            derive([[1]], [1]).evaluate([])

    def test_tokens_survive(self):
        for token in ['u', 'c', '[c]', '[x]', 'z', ':', '', '1.5', '-1', '1,2']:
            r = parse_count(token)
            self.assertIsNone(r.value)
            self.assertEqual(r.token, token)
        self.assertEqual(parse_count('0').value, 0)
        self.assertEqual(parse_count('1,234').value, 1234)


if __name__ == '__main__':
    unittest.main()
