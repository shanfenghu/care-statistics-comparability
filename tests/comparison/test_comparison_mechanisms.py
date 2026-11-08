"""Numerical and policy checks for test comparison mechanisms."""
import math
import unittest
from comparison_mechanisms import margin, interval_direction, query_geometry, transition


class MechanismChecks(unittest.TestCase):
    def test_repair_damage_and_ties_have_distinct_denominators(self):
        self.assertEqual(transition(1,-1,1),'repair')
        self.assertEqual(transition(1,1,-1),'damage')
        self.assertEqual(transition(1,0,1),'point_tie')
        self.assertEqual(transition(0,1,-1),'unresolved')
        self.assertEqual(transition(-1,-1,-1),'both_correct')
        self.assertEqual(transition(-1,1,1),'both_wrong')

    def test_multiplicity_can_reverse_an_event_order(self):
        d,u,s=query_geometry((1155,1105),(890,920))
        self.assertGreater(d,0);self.assertLess(d+u,0);self.assertEqual(s,-1)
        d,u,s=query_geometry((100,100),(90,80))
        self.assertEqual(d,0);self.assertGreater(u,0);self.assertEqual(s,1)
        d,u,s=query_geometry((150,100),(90,90))
        self.assertAlmostEqual(d+u,0);self.assertEqual(s,0)

    def test_strict_margin_and_interval_agree_with_clipped_lower_end(self):
        for pp in [(1.,20.),(100.,110.),(100.,100.),(50.,1.)]:
            for events in [(100.,1.),(5.,10.)]:
                score,direction=margin(pp,events)
                for q in [0.,.125,.5,1.,100.]:
                    self.assertEqual(direction if score>q else 0,interval_direction(pp,events,q))

    def test_exactly_representable_geographic_boundary_abstains(self):
        self.assertEqual(margin((100.,110.),(5.,5.)),(1.,-1))
        self.assertEqual(interval_direction((100.,110.),(5.,5.),1.),0)

    def test_nonrepresentable_float_boundary_is_not_a_real_identity_test(self):
        # Division rounds a real boundary. Preserve interval implementation semantics;
        # exact algebra alone does not specify equality at every binary64 boundary.
        score,d=margin((1.,20.),(5.,10.))
        self.assertEqual(score,19./15.)
        self.assertEqual(interval_direction((1.,20.),(5.,10.),score),-1)
        self.assertFalse(score>score)

    def test_temporal_anchor_boundary_and_lag_direction(self):
        for e,old_e,old_p in [(120,100,80),(90,100,80),(100,100,80)]:
            prediction=e*old_p/old_e
            score,d=margin((prediction,),(e,),old_p)
            self.assertEqual(d,(e>old_e)-(e<old_e))
            for q in [0.,score,score+.1]:
                self.assertEqual(d if score>q else 0,interval_direction((prediction,),(e,),q,old_p))

    def test_rounded_reference_requires_separation(self):
        self.assertEqual(transition(0,1,-1),'unresolved')
        with self.assertRaises(AssertionError):query_geometry((100,0),(90,80))


if __name__=='__main__':unittest.main()
