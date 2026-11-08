"""Post-prediction comparison corrections and crossed-margin diagnostics."""
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENUE = ROOT
DEST = VENUE / 'results/comparison-mechanism'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(x):
    return (x > 0) - (x < 0)


def margin(prediction, events, anchor=None):
    """Return unlabeled normalized margin and direction; anchor is prior native P."""
    if anchor is None:
        return abs(prediction[0] - prediction[1]) / sum(events), sign(prediction[0] - prediction[1])
    return abs(prediction[0] - anchor) / events[0], sign(prediction[0] - anchor)


def interval_direction(prediction, events, q, anchor=None):
    """Independent direct interval implementation, including strict boundaries."""
    if anchor is not None:
        low = max(0., prediction[0] - q * events[0]); high = prediction[0] + q * events[0]
        return 1 if low > anchor else -1 if high < anchor else 0
    lo = [max(0., p - q * e) for p, e in zip(prediction, events)]
    hi = [p + q * e for p, e in zip(prediction, events)]
    return 1 if lo[0] > hi[1] else -1 if hi[0] < lo[1] else 0


def query_geometry(events, people):
    """Separate event-size contrast from people-per-event heterogeneity.

    Exact integer products establish the native order independently of logs.
    Equal native counts are retained as ties, including any float cancellation.
    """
    ei, ej = events; pi, pj = people
    assert min(ei, ej, pi, pj) > 0
    d = math.log(ei / ej)
    u = math.log(pi / ei) - math.log(pj / ej)
    exact = sign(pi * ei * ej - pj * ej * ei)
    assert math.isclose(d + u, math.log(pi / pj), abs_tol=2e-14, rel_tol=2e-12)
    if pi != pj:
        assert sign(d + u) == exact
    return d, u, exact


def transition(reference, baseline, richer):
    if not reference:
        return 'unresolved'
    if not baseline or not richer:
        return 'point_tie'
    good_a, good_b = baseline == reference, richer == reference
    return ('both_correct' if good_b else 'damage') if good_a else ('repair' if good_b else 'both_wrong')


def analyze(case):
    temporal = case['task'] in ['temporal', 'asof']
    base = 'lagged_factor' if temporal else 'family_factor'
    methods = [base, 'context_forest']
    queries = []; rows = case['test']; checks = 0
    for f in range(2):
        pairs = [(i, None) for i in range(len(rows))] if temporal else itertools.combinations(range(len(rows)), 2)
        for i, j in pairs:
            x = rows[i]
            if temporal:
                es = (x['e'][f], x['previous']['e'][f]); ps = (x['p'][f], x['previous']['p'][f]); anchor = ps[1]
                radius = x['radius']
            else:
                y = rows[j]; es = (x['e'][f], y['e'][f]); ps = (x['p'][f], y['p'][f]); anchor = None
                radius = x['radius'] + y['radius']
            d, u, native = query_geometry(es, ps)
            reference = native if abs(ps[0] - ps[1]) > radius else 0
            rec = dict(family=f, i=i, j=j, reference=reference, event_direction=sign(es[0] - es[1]),
                       event_log_contrast=d, native_ratio_log_contrast=u, normalized_native_gap=abs(ps[0]-ps[1])/sum(ps),
                       methods={})
            for method in methods:
                m = case['methods'][method]; pred = m['test_prediction']
                pp = (pred[i][f],) if temporal else (pred[i][f], pred[j][f])
                ee = (es[0],) if temporal else es
                score, direction = margin(pp, ee, anchor)
                if method == base:
                    assert direction == sign(es[0] - es[1]), (case['name'], f, i, j)
                q = m['quantiles']['0.05']; assert q is not None and math.isfinite(q)
                own = direction if score > q else 0
                assert own == interval_direction(pp, ee, q, anchor)
                crosses = {}
                for threshold_method in methods:
                    qq = case['methods'][threshold_method]['quantiles']['0.05']
                    crosses[threshold_method] = direction if score > qq else 0
                    assert crosses[threshold_method] == interval_direction(pp, ee, qq, anchor)
                    checks += 1
                rec['methods'][method] = dict(point=direction, margin=score, guard=own, crossed=crosses)
            rec['transition'] = transition(reference, rec['methods'][base]['point'], rec['methods']['context_forest']['point'])
            queries.append(rec)
    # Recompute calibrated scores and replay every existing point/alpha=.05 count.
    for method in methods:
        m = case['methods'][method]
        for row, pred, score in zip(case['calibration'], m['calibration_prediction'], m['calibration_scores']):
            direct = max((abs(row['p'][f]-pred[f])+row['radius'])/row['e'][f] for f in range(2))
            assert math.isclose(direct, score, rel_tol=1e-12, abs_tol=1e-14)
        for saved in m['metrics']:
            if saved['policy'] not in ['point', 'alpha=0.05']:
                continue
            field = 'point' if saved['policy'] == 'point' else 'guard'
            rr = [r for r in queries if r['family'] == saved['family']]
            calc = dict(queries=len(rr), resolved=sum(r['reference'] != 0 for r in rr),
                        answered_all=sum(r['methods'][method][field] != 0 for r in rr),
                        answered=sum(r['reference'] != 0 and r['methods'][method][field] != 0 for r in rr),
                        wrong=sum(r['reference'] != 0 and r['methods'][method][field] not in [0, r['reference']] for r in rr))
            assert all(saved[k] == v for k, v in calc.items()), (case['name'], method, saved, calc)
    return dict(name=case['name'], group=case['group'], source=case['source'], task=case['task'],
                baseline=base, q={m:case['methods'][m]['quantiles']['0.05'] for m in methods},
                test_codes=[r['code'] for r in rows], queries=queries, interval_checks=checks)


def summarize(cases):
    base = cases[0]['baseline']; rr = [r for c in cases for r in c['queries']]; resolved = [r for r in rr if r['reference']]
    transitions = {k:sum(r['transition'] == k for r in rr) for k in ['both_correct','repair','damage','both_wrong','point_tie','unresolved']}
    near = {}
    for ratio in [1.05, 1.10, 1.25]:
        bins = {}
        for label, predicate in [('within',lambda d:abs(d)<=math.log(ratio)),('outside',lambda d:abs(d)>math.log(ratio))]:
            selected = [r for r in resolved if predicate(r['event_log_contrast'])]
            bins[label] = dict(queries=len(selected), baseline_ties=sum(r['methods'][base]['point']==0 for r in selected),
                baseline_errors=sum(r['methods'][base]['point'] not in [0,r['reference']] for r in selected),
                forest_errors=sum(r['methods']['context_forest']['point'] not in [0,r['reference']] for r in selected))
        near[str(ratio)] = bins
    crossed = []
    for margin_method, threshold_method in itertools.product([base,'context_forest'],repeat=2):
        dd = [r['methods'][margin_method]['crossed'][threshold_method] for r in resolved]
        crossed.append(dict(margin=margin_method,threshold=threshold_method,resolved=len(resolved),answered=sum(d!=0 for d in dd),
                            wrong=sum(d not in [0,r['reference']] for d,r in zip(dd,resolved))))
    return dict(queries=len(rr), resolved=len(resolved), transitions=transitions, event_ratio_screens=near,crossed_guards=crossed,
                q_by_case={c['name']:c['q'] for c in cases})


def main():
    source = VENUE/'results/comparison-robustness/summary.json'; summary=json.loads(source.read_text())
    DEST.mkdir(exist_ok=True); (DEST/'cases').mkdir(exist_ok=True)
    by_group = {}; records=[]; total_checks=0
    for row in summary['cases']:
        path=ROOT/row['path']; assert sha(path)==row['sha256']
        case=analyze(json.loads(path.read_text())); by_group.setdefault(case['group'],[]).append(case)
        out=DEST/'cases'/(case['name']+'.json');out.write_text(json.dumps(case,separators=(',',':'))+'\n')
        records.append(dict(path=str(out.relative_to(ROOT)),sha256=sha(out),input_path=row['path'],input_sha256=row['sha256']))
        total_checks+=case['interval_checks']
    result=dict(status='passed',analysis='Development-selected fixed-record mechanism diagnostics; no refitting or independent confirmation',
                script_sha256=sha(Path(__file__)),protocol_sha256=sha(VENUE/'experiments/comparison/config/mechanism-method.txt'),
                source_summary_sha256=sha(source),cases=records,groups={k:summarize(v) for k,v in by_group.items()},
                checks=dict(case_records=len(records),crossed_interval_equivalences=total_checks,existing_point_and_guard_counts_replayed=True,
                            calibration_scores_replayed=True,integer_log_identities=True),
                crossed_guard_assurance='Diagnostic only: swapping thresholds does not preserve conformal validity',
                uncertainty='Descriptive counts; prior paired authority-weight intervals and dependent split sensitivity retain their scope')
    (DEST/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checks=result['checks'],groups={k:v['transitions'] for k,v in result['groups'].items()}),indent=2))


if __name__=='__main__':
    main()
