"""Numerical and policy checks for verify comparison accounting."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENUE = ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect():
    """Verify all case/family point and own-guard balances, retaining ties.

    Reference-unresolved queries are inventoried but excluded from scored risks.
    Point ties and guard abstentions remain nonanswers, including when only one
    policy answers. Each risk uses its own positive answer denominator; an empty
    answer set has undefined risk. Exact fractions avoid a rounding tolerance.
    """
    summary_path = VENUE / 'results/comparison-mechanism/summary.json'
    summary = json.loads(summary_path.read_text())
    robust = json.loads((VENUE / 'results/comparison-robustness/summary.json').read_text())
    bindings = robust['raw_inputs'] + [dict(path=str(summary_path.relative_to(ROOT)), sha256=digest(summary_path))]
    for case in summary['cases']:
        bindings += [dict(path=case['path'], sha256=case['sha256']),
                     dict(path=case['input_path'], sha256=case['input_sha256'])]
    # Verify the entire fixed inventory before deriving any new accounting totals.
    for item in bindings:
        path = ROOT / item['path']
        assert path.is_file() and digest(path) == item['sha256'], item['path']
    totals = {}; checks = 0; query_inventory = 0
    for item in summary['cases']:
        case = json.loads((ROOT / item['path']).read_text())
        original = json.loads((ROOT / item['input_path']).read_text())
        assert case['name'] == original['name'] and case['group'] == original['group']
        base = case['baseline']; forest = 'context_forest'
        query_inventory += len(case['queries'])
        for family in (0, 1):
            for mode, saved_policy in [('point', 'point'), ('guard', 'alpha=0.05')]:
                c = Counter(queries=0, resolved=0, common=0, baseline_only=0,
                            forest_only=0, neither=0, repair=0, damage=0,
                            baseline_unique_wrong=0, forest_unique_wrong=0,
                            baseline_answered=0, forest_answered=0,
                            baseline_wrong=0, forest_wrong=0)
                for query in case['queries']:
                    if query['family'] != family:
                        continue
                    c['queries'] += 1
                    truth = query['reference']
                    if not truth:
                        continue
                    c['resolved'] += 1
                    b = query['methods'][base][mode]; f = query['methods'][forest][mode]
                    bw = bool(b and b != truth); fw = bool(f and f != truth)
                    c['baseline_answered'] += bool(b); c['forest_answered'] += bool(f)
                    c['baseline_wrong'] += bw; c['forest_wrong'] += fw
                    if b and f:
                        c['common'] += 1; c['repair'] += bw and not fw; c['damage'] += fw and not bw
                    elif b:
                        c['baseline_only'] += 1; c['baseline_unique_wrong'] += bw
                    elif f:
                        c['forest_only'] += 1; c['forest_unique_wrong'] += fw
                    else:
                        c['neither'] += 1
                assert sum(c[k] for k in ['common', 'baseline_only', 'forest_only', 'neither']) == c['resolved']
                assert c['forest_wrong'] - c['baseline_wrong'] == c['damage'] - c['repair'] + c['forest_unique_wrong'] - c['baseline_unique_wrong']
                assert c['forest_answered'] - c['baseline_answered'] == c['forest_only'] - c['baseline_only']
                for prefix, method in [('baseline', base), ('forest', forest)]:
                    saved = next(x for x in original['methods'][method]['metrics']
                                 if x['family'] == family and x['policy'] == saved_policy)
                    assert (c['queries'], c['resolved'], c[prefix+'_answered'], c[prefix+'_wrong']) == tuple(saved[k] for k in ['queries', 'resolved', 'answered', 'wrong'])
                    own = Fraction(c[prefix+'_wrong'], c[prefix+'_answered']) if c[prefix+'_answered'] else None
                    expected = Fraction(saved['wrong'], saved['answered']) if saved['answered'] else None
                    assert own == expected
                totals.setdefault(case['group']+':'+mode, Counter()).update(c)
                checks += 1
    return dict(scope='Retained-record accounting validation, not new fitting or independent evidence',
                verifier_sha256=digest(Path(__file__)), cases=len(summary['cases']),
                case_family_policy_checks=checks, retained_query_inventory=query_inventory,
                input_bindings=bindings, all_common_and_unique_error_balances_match=True,
                all_own_answer_counts_and_risks_match=True,
                groups={k:dict(v) for k,v in sorted(totals.items())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(inspect(), indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(result)
        print('Created', args.output)
    else:
        print(result, end='')
