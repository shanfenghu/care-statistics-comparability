"""Classical linear determinacy for exact count views over a finite atom space.

For an integer matrix A, rows denote published linear views of nonnegative
integer atom counts x. The query is q*x. A universal reconstruction exists iff
q lies in the rational row span of A. This is standard linear algebra, not a new
scientific result. Metadata must separately justify the common atom space.

The obstruction is universal: its pair of databases need not agree with any
particular observed release. Rounding, suppression, overlap across counting
units and unknown population coverage are not repaired by this module.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Sequence, Tuple, Union


@dataclass(frozen=True)
class Rewrite:
    """Rational coefficients w satisfying w*A=q, in input row order."""

    coefficients: Tuple[Fraction, ...]

    def evaluate(self, values: Sequence[int]) -> Fraction:
        """Apply the rewrite to exact view values, after semantic validation.

        This method makes no assertion that supplied values are exact,
        unsuppressed, mutually consistent, or even drawn from the same cohort.
        """
        if len(values) != len(self.coefficients):
            raise ValueError("One value is required for every view row")
        return sum((w * v for w, v in zip(self.coefficients, values)), Fraction(0))


@dataclass(frozen=True)
class Obstruction:
    """Two nonnegative integer atom vectors with equal views and unequal query."""

    left: Tuple[int, ...]
    right: Tuple[int, ...]


def _rref(rows, n_columns):
    """Return exact reduced echelon rows and pivots, leaving augmented columns."""
    matrix = [[Fraction(v) for v in row] for row in rows]
    pivot_columns = []
    for column in range(n_columns):
        pivot = next((i for i in range(len(pivot_columns), len(matrix))
                      if matrix[i][column]), None)
        if pivot is None:
            continue
        k = len(pivot_columns)
        matrix[k], matrix[pivot] = matrix[pivot], matrix[k]
        scale = matrix[k][column]
        matrix[k] = [v / scale for v in matrix[k]]
        for i in range(len(matrix)):
            if i != k and matrix[i][column]:
                scale = matrix[i][column]
                matrix[i] = [v - scale * p for v, p in zip(matrix[i], matrix[k])]
        pivot_columns.append(column)
        if len(pivot_columns) == len(matrix):
            break
    return matrix, pivot_columns


def _lcm(a, b):
    return abs(a * b) // gcd(a, b)


def derive(views: Sequence[Sequence[int]], query: Sequence[int]) -> Union[Rewrite, Obstruction]:
    """Derive a universal exact reconstruction or a verifiable obstruction.

    Example: derive([[1, 1], [0, 1]], [1, 0]) returns coefficients (1,-1).
    derive([[1,1]], [1,0]) instead returns indistinguishable atom counts.

    Inputs are integers; a nonempty atom space is required, but an empty view
    family is allowed. Negative coefficients are accepted for count contrasts.
    The algorithm uses rational Gaussian elimination, intended for small
    structural models rather than high-dimensional database workloads.
    """
    n = len(query)
    if not n or any(len(row) != n for row in views):
        raise ValueError("A nonempty common atom space and rectangular views are required")
    flat = list(query) + [v for row in views for v in row]
    if any(not isinstance(v, int) or isinstance(v, bool) for v in flat):
        raise TypeError("Integer structural coefficients are required")
    m = len(views)
    # Solving A^T*w=q constructs the rewriting without a floating tolerance.
    augmented = [[views[i][j] for i in range(m)] + [query[j]] for j in range(n)]
    echelon, pivots = _rref(augmented, m)
    inconsistent = any(not any(row[:m]) and row[m] for row in echelon)
    if not inconsistent:
        weights = [Fraction(0)] * m
        for i, pivot in enumerate(pivots):
            weights[pivot] = echelon[i][m]
        return Rewrite(tuple(weights))

    # A null direction with q*h != 0 is a finite, independently checkable proof.
    echelon, pivots = _rref(views, n)
    for free in (j for j in range(n) if j not in pivots):
        direction = [Fraction(0)] * n
        direction[free] = Fraction(1)
        for i, pivot in enumerate(pivots):
            direction[pivot] = -echelon[i][free]
        if sum(q * h for q, h in zip(query, direction)):
            scale = reduce(_lcm, (v.denominator for v in direction), 1)
            h = [int(v * scale) for v in direction]
            return Obstruction(tuple(max(v, 0) for v in h),
                               tuple(max(-v, 0) for v in h))
    raise AssertionError("Inconsistent transpose system must have a separating null direction")
