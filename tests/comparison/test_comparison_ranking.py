"""Numerical and policy checks for test comparison ranking."""
from copy import deepcopy
import unittest
import numpy as np
from scipy.optimize import check_grad
from comparison_ranking import logistic_objective, fit_pairs, prepare, pair_data, score_queries


class RankingTests(unittest.TestCase):
    def test_gradient_and_extreme_margins(self):
        x = np.array([[1.,2.],[-2.,3.],[4.,-1.]])
        y = np.array([1.,-1.,1.]); w = np.array([.2,-.3])
        error = check_grad(lambda v:logistic_objective(v,x,y,.1)[0],
                           lambda v:logistic_objective(v,x,y,.1)[1],w)
        self.assertLess(error, 1e-6)
        margin = y*np.sum(x*w,axis=1)
        expected_loss = np.logaddexp(0,-margin).mean()+.1*np.sum(w*w)/2
        self.assertAlmostEqual(logistic_objective(w,x,y,.1)[0],expected_loss)
        for v in [np.array([1000.,-1000.]),np.array([-1000.,1000.])]:
            loss, gradient = logistic_objective(v,x,y,.1)
            self.assertTrue(np.isfinite(loss) and np.isfinite(gradient).all())

    def test_optimizer_recovers_transitive_synthetic_order(self):
        x = np.arange(8,dtype=float)[:,None]
        rows = [dict(p=[10+i,10+i],radius=0) for i in range(8)]
        difference,direction = pair_data(rows,x,0)
        w,status = fit_pairs(difference,direction,.01,dict(optimizer=dict(maxiter=2000,ftol=1e-12,gtol=1e-9)))
        self.assertGreater(w[0],0)
        np.testing.assert_array_equal(np.sign(difference@w),direction)
        self.assertTrue(status['success'])

    def test_native_overlap_not_given_arbitrary_training_label(self):
        rows = [dict(p=[10,10],radius=2),dict(p=[13,13],radius=2),dict(p=[20,20],radius=2)]
        difference,direction = pair_data(rows,np.arange(3)[:,None],0)
        self.assertEqual(len(direction),2)
        np.testing.assert_array_equal(direction,[-1,-1])

    def test_evaluation_targets_do_not_change_features_scores_or_selection(self):
        train = [dict(e=[100+10*i,200+7*i],p=[80+i,90+i],region='a',radius=0) for i in range(8)]
        test = deepcopy(train[:4]); altered = deepcopy(test)
        for r in altered:r['p']=[999999,1]
        x,xx,_,_ = prepare(train,test,0)
        x2,xx2,_,_ = prepare(train,altered,0)
        np.testing.assert_array_equal(x,x2);np.testing.assert_array_equal(xx,xx2)
        scores=xx@np.arange(xx.shape[1])
        a=score_queries(test,scores,0,.75);b=score_queries(altered,scores,0,.75)
        np.testing.assert_array_equal(a['point'],b['point'])
        np.testing.assert_array_equal(a['budget'],b['budget'])


if __name__ == '__main__':
    unittest.main()
