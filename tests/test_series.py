"""Independent finite-state references and unbounded counting countermodels."""
from collections import deque
from fractions import Fraction
import itertools
import random
import unittest

from countviews import Rewrite
from countviews.linear import derive
from countviews.series import (PopulationWitness, compile_direct, compile_refine,
                              compile_batch_direct, compile_workload, compile_vector,
                              counter, indicator, restrict, verify_witness, Machine)


def product_samples(tables, initials):
    """Enumerate the full reachable DFA product, without weighted saturation."""
    queue=deque([(tuple(initials),())]);seen={tuple(initials)};words=[]
    while queue:
        state,word=queue.popleft();words.append(word)
        for i,symbol in enumerate(('a','b')):
            new=tuple(t[s][i] for t,s in zip(tables,state))
            if new not in seen:seen.add(new);queue.append((new,word+(symbol,)))
    return words


class SeriesTests(unittest.TestCase):
    def test_random_dfa_families_against_full_product(self):
        rng=random.Random(3107)
        for _ in range(100):
            tables=[[[rng.randrange(3),rng.randrange(3)] for _ in range(3)] for _ in range(4)]
            machines=[indicator(('a','b'),t,0,{s for s in range(3) if rng.randrange(2)}) for t in tables]
            words=product_samples(tables,[0]*4)
            profiles=[[m.evaluate(w) for w in words] for m in machines]
            reference=derive(profiles[:-1],profiles[-1])
            for result in [compile_direct(machines[:-1],machines[-1]),
                           compile_refine(machines[:-1],machines[-1]),
                           compile_vector(machines[:-1],[machines[-1]])[0]]:
                self.assertEqual(isinstance(result.result,Rewrite),isinstance(reference,Rewrite))
                if isinstance(result.result,Rewrite):
                    for w in words:
                        self.assertEqual(result.result.evaluate([m.evaluate(w) for m in machines[:-1]]),machines[-1].evaluate(w))
                else:
                    self.assertTrue(verify_witness(machines[:-1],machines[-1],result.result))
                    self.assertLessEqual(sum(bool(l or r) for l,r in zip(result.result.left,result.result.right)),4)
            self.assertLessEqual(compile_refine(machines[:-1],machines[-1]).oracle_calls,4)

    def test_unbounded_event_counts_do_not_determine_people(self):
        events=counter(('a','b'),{'a':1})
        people=indicator(('a','b'),[[1,0],[1,1]],0,{1})
        for compiler in [compile_direct,compile_refine]:
            r=compiler([events],people)
            self.assertIsInstance(r.result,PopulationWitness)
            self.assertTrue(verify_witness([events],people,r.result))
        self.assertEqual(events.evaluate(('a',)*150),150)
        self.assertEqual(people.evaluate(('a',)*150),1)

    def test_nonoverlap_domain_proves_stock_flow_but_not_dedup(self):
        alphabet=('start','end')
        # No overlapping episodes: closed, open, invalid.
        domain=indicator(alphabet,[[1,2],[2,0],[2,2]],0,{0,1})
        starts=counter(alphabet,{'start':1});ends=counter(alphabet,{'end':1})
        open_end=indicator(alphabet,[[1,0],[1,0]],0,{1})
        people=indicator(alphabet,[[1,0],[1,1]],0,{1})
        for compiler in [compile_direct,compile_refine]:
            r=compiler([starts,ends],open_end,domain)
            self.assertIsInstance(r.result,Rewrite)
            self.assertEqual(r.result.coefficients,(1,-1))
            bad=compiler([starts],people,domain)
            self.assertIsInstance(bad.result,PopulationWitness)
            self.assertTrue(verify_witness([starts],people,bad.result,domain))

    def test_refinement_seeds_and_empty_domain(self):
        alphabet=('a','b')
        a=counter(alphabet,{'a':1});b=counter(alphabet,{'b':1})
        total=counter(alphabet,{'a':1,'b':1})
        r=compile_refine([a,b],total,seeds=[('a',),('b',)])
        self.assertEqual(r.result.coefficients,(1,1));self.assertEqual(r.oracle_calls,1)
        empty=indicator(alphabet,[[0,0]],0,set())
        self.assertIsInstance(compile_refine([],total,empty).result,Rewrite)

    def test_illegal_witness_is_rejected(self):
        domain=indicator(('a','b'),[[1,0],[1,1]],0,{0})
        events=counter(('a','b'),{'a':1})
        bad=PopulationWitness((('a',),),(1,),(0,))
        self.assertFalse(verify_witness([],events,bad,domain))

    def test_indicator_rejects_noninteger_state_indices(self):
        for bad in (Fraction(1,2), 0.5, True, '0'):
            with self.subTest(role='initial', value=bad):
                with self.assertRaises(TypeError):indicator(('a',),[[0]],bad,{0})
            with self.subTest(role='accepting', value=bad):
                with self.assertRaises(TypeError):indicator(('a',),[[0]],0,{bad})
            with self.subTest(role='transition', value=bad):
                with self.assertRaises(TypeError):indicator(('a',),[[bad]],0,{0})
        for transitions,initial,accepting in [([],0,set()),([[0,0]],0,{0}),
                                              ([[0]],-1,{0}),([[0]],0,{1})]:
            with self.assertRaises(ValueError):indicator(('a',),transitions,initial,accepting)
        valid=indicator(('a',),[[0]],0,{0})
        self.assertEqual(valid.evaluate(()),1)
        self.assertEqual(valid.evaluate(('a',)*10),1)

    def test_witness_requires_nonnegative_integer_multiplicities(self):
        events=counter(('a',),{'a':1})
        for bad in (Fraction(1,2), True, -1, 1.0):
            witness=PopulationWitness((('a',),),(bad,),(0,))
            self.assertFalse(verify_witness([],events,witness))
        witness=PopulationWitness((('a',),),(1,),(0,))
        self.assertTrue(verify_witness([],events,witness))

    def test_direct_machine_rejects_noninteger_destinations(self):
        for bad in (Fraction(1,2), 0.0, True):
            with self.assertRaises(TypeError):
                Machine(('a',),(1,),{'a':(((bad,1),),)},(1,))

    def test_shared_structure_preserves_different_terminal_outputs(self):
        alpha=('a','b')
        yes=indicator(alpha,[[1,0],[1,1]],0,{1})
        no=indicator(alpha,[[1,0],[1,1]],0,{0})
        total=indicator(alpha,[[1,0],[1,1]],0,{0,1})
        for r in compile_batch_direct([yes,no],[total,yes]):
            self.assertIsInstance(r.result,Rewrite)
        r=compile_workload([yes,no],[total,yes])
        self.assertEqual(r[0].result.coefficients,(1,1))
        signed=Machine(yes.alphabet,yes.initial,yes.transitions,tuple(-v for v in yes.terminal))
        self.assertEqual(compile_direct([yes],signed).result.coefficients,(-1,))

    def test_non_deterministic_domain_is_rejected(self):
        event=counter(('a','b'),{'a':1})
        with self.assertRaises(ValueError):restrict(event,event)

    def test_batch_refinement_bounds_and_all_targets_checked(self):
        alpha=('a','b')
        a=counter(alpha,{'a':1});b=counter(alpha,{'b':1})
        total=counter(alpha,{'a':1,'b':1})
        people=indicator(alpha,[[1,0],[1,1]],0,{1})
        queries=[total,a,b,people]
        for compiler in [compile_workload,compile_vector]:
            results=compiler([a,b],queries)
            self.assertEqual([isinstance(r.result,Rewrite) for r in results],[True,True,True,False])
            for query,r in zip(queries,results):
                if isinstance(r.result,Rewrite):
                    for n in range(5):
                        for word in itertools.product(alpha,repeat=n):
                            self.assertEqual(r.result.evaluate([a.evaluate(word),b.evaluate(word)]),query.evaluate(word))
                else:self.assertTrue(verify_witness([a,b],query,r.result))
            self.assertLessEqual(sum(r.oracle_calls for r in results),4+4)
        positive=compile_vector([a,b],[total,a,b]*10)
        self.assertLessEqual(sum(r.oracle_calls for r in positive),3)
        self.assertEqual(compile_vector([a,b],[]),())

    def test_seed_histories_are_reevaluated_after_definition_change(self):
        alpha=('a','b')
        old=counter(alpha,{'a':1});new=counter(alpha,{'a':1,'b':1})
        first=compile_refine([old],old)
        self.assertIsInstance(first.result,Rewrite)
        changed=compile_refine([new],old,seeds=first.samples)
        self.assertIsInstance(changed.result,PopulationWitness)
        self.assertTrue(verify_witness([new],old,changed.result))
        self.assertLessEqual(changed.oracle_calls,2)

    def test_obstruction_support_bound_is_attained(self):
        alpha=('a','b','c','z')
        def accepts(letters):
            return indicator(alpha,[[1 if a in letters else 2 for a in alpha],
                                    [3]*4,[3]*4,[3]*4],0,{1})
        domain=indicator(alpha,[[1]*4,[2]*4,[2]*4],0,{1})
        views=[accepts({a,'z'}) for a in alpha[:-1]];target=accepts({'z'})
        result=compile_refine(views,target,domain).result
        self.assertTrue(verify_witness(views,target,result,domain))
        self.assertEqual(sum(bool(a or b) for a,b in zip(result.left,result.right)),4)
        self.assertTrue(all(len(w)==1 for w,a,b in zip(result.words,result.left,result.right) if a or b))


if __name__=='__main__':unittest.main()
