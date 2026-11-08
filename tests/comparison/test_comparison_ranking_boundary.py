"""Numerical and policy checks for test comparison ranking boundary."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np
from comparison_ranking_boundary import extend_selection,score_rows,old_new_contrasts


class BoundaryTests(unittest.TestCase):
    def test_validation_tie_preserves_stronger_original_fit(self):
        train=[dict(code=str(i),e=[100+i,200+i],p=[80+i,150+i],region='a',radius=0) for i in range(9)]
        old=dict(selected_penalty=.1,inner_candidates=[dict(penalty=.1,loss=0.,resolved=9,wrong_plus_half_ties=0.,folds=[])],
                 coefficients=[3.],mean=[0.],scale=[1.],feature_names=['log_events'],optimizer=dict(success=True))
        cfg=dict(new_regularization=[.001,.0001],optimizer=dict(maxiter=5000,maxls=200,ftol=1e-12,gtol=1e-9))
        with patch('comparison_ranking_boundary.partition',side_effect=lambda code,label:int(code)%3):
            r=extend_selection(train,0,'test',old,cfg)
        self.assertTrue(r['reused_original_final_fit'])
        self.assertEqual(r['fit']['coefficients'],old['coefficients'])
        self.assertEqual(len(r['fit']['inner_candidates']),3)

    def test_outer_target_changes_do_not_change_scores(self):
        rows=[dict(e=[100+i,200+i],p=[20,30],region='a') for i in range(4)]
        other=deepcopy(rows)
        for r in other:r['p']=[1,999999]
        fit=dict(feature_names=['log_events','other_event_ratio','region=a'],mean=[0.,0.,0.],scale=[1.,1.,1.],coefficients=[2.,3.,4.])
        np.testing.assert_array_equal(score_rows(rows,fit,0),score_rows(other,fit,0))

    def test_risk_contrast_retains_each_answer_set(self):
        # Unit endpoint weights expose the difference between own-set ratios
        # (1/2 - 2/3) and the common-answer-set comparison (1/2 - 2/2).
        class UnitWeights:
            def multinomial(self,*args):return np.ones(3,dtype=int)
        q=dict(a=[0,0,1],b=[1,2,2],truth=[1,1,1],methods={})
        for name,answer in [('pairwise_logistic',[1,0,-1]),('pairwise_original',[-1,1,-1])]:
            q['methods'][name]={p:answer for p in ['point','budget']}
        cases=[dict(test=[dict(code=str(i)) for i in range(3)],queries=[q])]
        with patch('comparison_ranking_boundary.np.random.default_rng',return_value=UnitWeights()):
            result=old_new_contrasts(cases,dict(seed=1,bootstrap_authority_repetitions=2))
        for interval in result.values():np.testing.assert_allclose(interval,[-100/6,-100/6])


if __name__=='__main__':unittest.main()
