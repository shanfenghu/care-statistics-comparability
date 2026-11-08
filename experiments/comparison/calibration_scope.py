"""Joint, specified-family and Bonferroni calibration replay on fixed predictions."""
from collections import defaultdict
import json
import math
import platform
from pathlib import Path

import numpy as np

from comparison_panel import ROOT, conformal_quantile, digest, order
from comparison_robustness import counts, queries, rates

VENUE = ROOT
OUT = VENUE / 'results/calibration-scope'
PROTOCOL = VENUE / 'experiments/comparison/config/calibration-method.txt'
SCOPES = ('vector', 'specified-family', 'Bonferroni-vector')


def thresholds(scores, alpha, scope):
    """Return two radii; infinity is a legitimate finite-rank outcome.

    Rows are calibration authorities, columns are the two fixed target families.
    Bonferroni retains joint-family nominal coverage, unlike specified-family.
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or scores.shape[1] != 2 or not len(scores):
        raise ValueError('Expected a nonempty authority-by-two-family matrix')
    if not np.isfinite(scores).all() or (scores < 0).any() or not 0 < alpha < 1:
        raise ValueError('Scores must be finite/nonnegative and alpha in (0,1)')
    if scope == 'vector':
        return np.repeat(conformal_quantile(scores.max(axis=1), alpha), 2)
    if scope not in SCOPES:
        raise ValueError('Unknown calibration scope')
    level = alpha / 2 if scope == 'Bonferroni-vector' else alpha
    return np.array([conformal_quantile(scores[:, f], level) for f in (0, 1)])


def scalar_answer(pred, events, qs, a, b, family, anchor=None):
    """Independent scalar endpoint replay, preserving strict floating comparisons."""
    lo = max(0., float(pred[a, family]) - float(qs[family]) * float(events[a, family]))
    hi = float(pred[a, family]) + float(qs[family]) * float(events[a, family])
    if anchor is None:
        other_lo = max(0., float(pred[b, family]) - float(qs[family]) * float(events[b, family]))
        other_hi = float(pred[b, family]) + float(qs[family]) * float(events[b, family])
    else:
        other_lo = other_hi = float(anchor)
    return 1 if lo > other_hi else -1 if hi < other_lo else 0


def evaluate(case):
    """Return policy records plus verification totals for one unchanged fitted case."""
    cal, test = case['calibration'], case['test']
    ep = np.array([r['e'] for r in cal], float)
    native = np.array([r['p'] for r in cal], float)
    rad = np.array([r['radius'] for r in cal], float)[:, None]
    events = np.array([r['e'] for r in test], float)
    target = np.array([r['p'] for r in test], float)
    test_rad = np.array([r['radius'] for r in test], float)[:, None]
    records = []
    checks = defaultdict(int)
    for method, saved in case['methods'].items():
        cp = np.array(saved['calibration_prediction'], float)
        tp = np.array(saved['test_prediction'], float)
        scores = (np.abs(native - cp) + rad) / ep
        np.testing.assert_array_equal(scores.max(axis=1), saved['calibration_scores'])
        qq = queries(test, tp, case['task'] != 'geographic')
        for alpha in (.025, .05, .1):
            original = {}
            for scope in SCOPES:
                qs = thresholds(scores, alpha, scope)
                if scope == 'vector':
                    old = saved['quantiles'][str(alpha)]
                    assert (old is None and np.isinf(qs).all()) or np.all(qs == old)
                lo = np.maximum(0, tp - events * qs)
                hi = tp + events * qs
                cover = (lo <= target - test_rad) & (hi >= target + test_rad)
                for query in qq:
                    f, a, b = query['f'], query['a'], query['b']
                    answer = (order(lo[:, f], hi[:, f], query['anchor'], query['anchor'])
                              if query['anchor'] is not None else
                              order(lo[a, f], hi[a, f], lo[b, f], hi[b, f]))
                    scalar = np.array([scalar_answer(tp, events, qs, int(i), int(j), f,
                                        None if query['anchor'] is None else query['anchor'][z])
                                       for z, (i, j) in enumerate(zip(a, b))])
                    np.testing.assert_array_equal(answer, scalar)
                    checks['scalar_decisions'] += len(answer)
                    result = counts(query['truth'], answer)
                    if scope == 'vector':
                        original[f] = answer
                        old = next(v for v in saved['metrics'] if v['family'] == f
                                   and v['policy'] == 'alpha=' + str(alpha))
                        for k, v in result.items():
                            assert v == old[k], (case['name'], method, alpha, f, k)
                        checks['old_family_policies'] += 1
                    base = original[f]
                    resolved = query['truth'] != 0
                    added = resolved & (base == 0) & (answer != 0)
                    removed = resolved & (base != 0) & (answer == 0)
                    if scope == 'specified-family':
                        assert np.all(qs <= thresholds(scores, alpha, 'vector'))
                        assert np.all(answer[base != 0] == base[base != 0])
                        checks['nested_family_policies'] += 1
                    result.update(added=int(added.sum()),
                                  added_wrong=int((added & (answer != query['truth'])).sum()),
                                  removed=int(removed.sum()),
                                  removed_wrong=int((removed & (base != query['truth'])).sum()),
                                  rows=len(test), covered_rows=int(cover[:, f].sum()),
                                  covered_vectors=int(cover.all(axis=1).sum()),
                                  infinite=int(not math.isfinite(qs[f])))
                    records.append(dict(case=case['name'], group=case['group'],
                                        task=case['task'], method=method, alpha=alpha,
                                        scope=scope, family=f, n=len(cal),
                                        quantile=None if not math.isfinite(qs[f]) else float(qs[f]),
                                        **result))
    return records, dict(checks)


def main():
    """Verify the pinned inputs, then write all outcomes without ranking selection."""
    source = VENUE / 'results/comparison-robustness/summary.json'
    source_summary = json.loads(source.read_text())
    records = []
    checks = defaultdict(int)
    for item in source_summary['cases']:
        path = ROOT / item['path']
        assert digest(path) == item['sha256']
        rr, cc = evaluate(json.loads(path.read_text()))
        records.extend(rr)
        for key, value in cc.items():
            checks[key] += value
    totals = defaultdict(lambda: defaultdict(int))
    for row in records:
        for family in (str(row['family']), 'all'):
            key = (row['group'], row['method'], row['alpha'], row['scope'], family)
            for field in ('queries', 'resolved', 'answered_all', 'answered', 'wrong',
                          'added', 'added_wrong', 'removed', 'removed_wrong', 'rows',
                          'covered_rows', 'covered_vectors', 'infinite'):
                totals[key][field] += row[field]
    summaries = []
    for (group, method, alpha, scope, family), values in sorted(totals.items()):
        summaries.append(dict(group=group, method=method, alpha=alpha, scope=scope,
                              family=family, **rates(values),
                              row_bin_coverage=values['covered_rows']/values['rows'],
                              vector_coverage=values['covered_vectors']/values['rows']))
    OUT.mkdir(parents=True, exist_ok=True)
    detail = OUT / 'policies.json'
    detail.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    summary = dict(protocol_sha256=digest(PROTOCOL), script_sha256=digest(__file__),
                   source_summary_sha256=digest(source), source_cases=source_summary['cases'],
                   policies_sha256=digest(detail), environment=dict(python=platform.python_version(),
                   numpy=np.__version__), cases=len(source_summary['cases']),
                   verification=dict(checks), summaries=summaries,
                   status='Post-panel development; fixed predictions; no independent confirmation')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(cases=summary['cases'], records=len(records), **checks)))


if __name__ == '__main__':
    main()
