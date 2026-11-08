"""Scientific plot visibility when a policy has no defined error rate."""
import math
import unittest

import matplotlib.pyplot as plt
import numpy as np

from render_results import set_risk_axis_limits


class RiskAxisTests(unittest.TestCase):
    def test_undefined_first_policy_keeps_other_errors_inside_the_axis(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        ax.plot([20, 50, 80], [np.nan, 0., 0.])
        ax.plot([20, 50, 80], [.1, .2, .3])
        set_risk_axis_limits(ax)
        lower, upper = ax.get_ylim()
        self.assertEqual(lower, 0)
        self.assertGreater(upper, .3)
        self.assertTrue(math.isfinite(upper))

    def test_entirely_undefined_policies_keep_a_finite_positive_axis(self):
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        ax.plot([20, 50, 80], [np.nan, np.nan, np.nan])
        set_risk_axis_limits(ax)
        lower, upper = ax.get_ylim()
        self.assertEqual(lower, 0)
        self.assertTrue(math.isfinite(upper))
        self.assertGreater(upper, lower)


if __name__ == '__main__':
    unittest.main()
