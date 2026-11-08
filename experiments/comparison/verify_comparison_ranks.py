"""Numerical and policy checks for verify comparison ranks."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENUE = ROOT
ALPHAS = (Fraction(1, 40), Fraction(1, 20), Fraction(1, 10))


def rank_bounds(n, alpha):
    """Return k, single failure and joint-pair failure upper coefficients.

    n is a positive calibration size and alpha an exact Fraction in (0,1).
    The predictor is fixed. The pair coefficient requires joint exchangeability
    of n calibration plus two new vectors; two separate marginal assurances
    alone do not imply it. k=n+1 denotes an infinite quantile and zero failures.
    No gamma or selected-risk probability is estimated by this function.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(alpha, Fraction) or not 0 < alpha < 1:
        raise ValueError('alpha must be an exact Fraction in (0,1)')
    k = math.ceil((n+1)*(1-alpha))
    single = Fraction(n+1-k, n+1)
    pair = 1-Fraction(k*(k+1), (n+1)*(n+2))
    return k, single, pair


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect():
    """Verify the entire retained inventory before deriving case coefficients."""
    summary_path = VENUE / 'results/comparison-robustness/summary.json'
    summary = json.loads(summary_path.read_text())
    bindings = [dict(path=str(summary_path.relative_to(ROOT)), sha256=digest(summary_path))]
    bindings += summary['raw_inputs'] + summary['cases']
    for item in bindings:
        path = ROOT / item['path']
        if path.is_symlink() or ROOT not in path.resolve().parents:
            raise ValueError('input must remain inside the project')
        if digest(path) != item['sha256']:
            raise ValueError('input checksum differs: '+item['path'])
    records = []; grouped = defaultdict(list)
    for item in summary['cases']:
        case = json.loads((ROOT / item['path']).read_text())
        n = len(case['calibration'])
        for alpha in ALPHAS:
            k, single, pair = rank_bounds(n, alpha)
            assert k == math.ceil((n+1)*(1-float(alpha)))
            assert 0 <= pair <= 2*single <= 2*alpha
            for method in case['methods'].values():
                assert len(method['calibration_scores']) == n
                assert (method['quantiles'][str(float(alpha))] is None) == (k > n)
            record = dict(case=case['name'], source=case['source'], task=case['task'],
                          n=n, alpha=str(alpha), k=k, infinite_quantile=k > n,
                          single_failure_coefficient=str(single),
                          pair_failure_coefficient=str(pair), coarse_pair=str(2*alpha),
                          pair_scope=('geographic pair' if case['task'] == 'geographic'
                                      else 'hypothetical two new vectors, not the known-anchor temporal query'))
            records.append(record)
            grouped[(case['source'], case['task'], str(alpha))].append(record)
    groups = []
    for (source, task, alpha), rows in sorted(grouped.items()):
        finite = [r for r in rows if not r['infinite_quantile']]
        groups.append(dict(source=source, task=task, alpha=alpha, cases=len(rows),
                           n=sorted({r['n'] for r in rows}),
                           infinite_quantile_cases=len(rows)-len(finite),
                           finite_pair_coefficient_range=(
                               [str(min(Fraction(r['pair_failure_coefficient']) for r in finite)),
                                str(max(Fraction(r['pair_failure_coefficient']) for r in finite))]
                               if finite else None)))
    return dict(scope='Retained-parameter audit under hypothetical exchangeability; no new predictive evidence',
                verifier_sha256=digest(Path(__file__)), input_bindings=bindings,
                cases=len(summary['cases']), parameter_settings=len(records),
                all_configured_floating_ranks_and_infinite_statuses_match=True,
                gamma_estimated=False, groups=groups, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(inspect(), indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(result)
    else:
        print(result, end='')
