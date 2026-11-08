"""Pairwise logistic order training with authority-grouped inner validation."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import platform
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from comparison_panel import ROOT, CONFIG, digest, features, load_rows, order
from comparison_robustness import RCFG, counts, partition, queries, rates, top_budget

CFG = ROOT / 'experiments/comparison/config/ranking.json'
PROTOCOL = ROOT / 'experiments/comparison/config/ranking-method.txt'
OUT = ROOT / 'results/comparison-ranking-stable'
BASE = ROOT / 'results/comparison-robustness'
METHODS = ('family_factor', 'context_ridge', 'context_forest', 'pairwise_logistic')


def pair_data(rows, x, family):
    """One orientation per native-bin-resolved pair; no arbitrary tie labels."""
    a, b = np.array(list(combinations(range(len(rows)), 2)), int).T
    p = np.array([r['p'][family] for r in rows], float)
    radius = np.array([r['radius'] for r in rows], float)
    y = order(p[a]-radius[a], p[a]+radius[a], p[b]-radius[b], p[b]+radius[b])
    use = y != 0
    return x[a[use]]-x[b[use]], y[use].astype(float)


def logistic_objective(w, difference, direction, penalty):
    """Mean pair loss and analytic gradient, with stable logistic arithmetic."""
    # Explicit contractions avoid the platform BLAS floating-status warnings;
    # they also make the small-dimensional optimisation easy to check by sums.
    margin = direction * np.einsum('ij,j->i', difference, w, optimize=False)
    value = np.logaddexp(0, -margin).mean() + penalty * np.dot(w, w) / 2
    gradient = np.einsum('ij,i->j', difference, -direction * expit(-margin), optimize=False) / len(direction) + penalty*w
    return value, gradient


def fit_pairs(difference, direction, penalty, cfg):
    """Solve a convex penalised ranking problem; fail before scoring on bad fits."""
    assert len(direction) and penalty > 0
    fit = minimize(logistic_objective, np.zeros(difference.shape[1]),
                   args=(difference, direction, penalty), jac=True,
                   method='L-BFGS-B', options=cfg['optimizer'])
    assert fit.success and np.isfinite(fit.x).all(), str(fit.message)
    return fit.x, dict(success=bool(fit.success), iterations=int(fit.nit),
                       message=str(fit.message), objective=float(fit.fun),
                       gradient_inf=float(np.max(np.abs(fit.jac))))


def prepare(train, evaluate, family):
    """Fit feature representation on training rows, with no evaluated targets."""
    vector = DictVectorizer(sparse=False)
    scale = StandardScaler()
    x = scale.fit_transform(vector.fit_transform([features(r, family) for r in train]))
    xx = scale.transform(vector.transform([features(r, family) for r in evaluate]))
    return x, xx, vector, scale


def fit_ranker(train, evaluate, family, label, cfg):
    """Choose regularisation by whole-authority inner folds, then refit once."""
    inner_ids = sorted({partition(r['code'], label) for r in train})
    assert len(inner_ids) == 3
    inner = []
    for fold_id in inner_ids:
        fitting = [r for r in train if partition(r['code'], label) != fold_id]
        validating = [r for r in train if partition(r['code'], label) == fold_id]
        assert not ({r['code'] for r in fitting} & {r['code'] for r in validating})
        x, xx, _, _ = prepare(fitting, validating, family)
        difference, direction = pair_data(fitting, x, family)
        vd, vy = pair_data(validating, xx, family)
        inner.append((fold_id, difference, direction, vd, vy))
    candidates = []
    for penalty in cfg['regularization']:
        wrong_half_ties, resolved = 0., 0
        checks = []
        for fold_id, difference, direction, vd, vy in inner:
            w, status = fit_pairs(difference, direction, penalty, cfg)
            answer = np.sign(np.einsum('ij,j->i', vd, w, optimize=False))
            loss = float(np.sum((answer != vy) & (answer != 0)) + .5*np.sum(answer == 0))
            wrong_half_ties += loss
            resolved += len(vy)
            checks.append(dict(fold=fold_id, resolved=len(vy), wrong_plus_half_ties=loss,
                               training_pairs=len(direction), optimizer=status))
        candidates.append(dict(penalty=penalty, loss=wrong_half_ties/resolved,
                               resolved=resolved, wrong_plus_half_ties=wrong_half_ties, folds=checks))
    selected = min(candidates, key=lambda c: (c['loss'], -c['penalty']))['penalty']
    x, xx, vector, scale = prepare(train, evaluate, family)
    difference, direction = pair_data(train, x, family)
    w, status = fit_pairs(difference, direction, selected, cfg)
    record = dict(selected_penalty=selected, inner_candidates=candidates,
                  training_pairs=len(direction), feature_names=vector.get_feature_names_out().tolist(),
                  mean=scale.mean_.tolist(), scale=scale.scale_.tolist(), coefficients=w.tolist(),
                  optimizer=status)
    return np.einsum('ij,j->i', xx, w, optimize=False), record


def score_queries(test, scores, family, budget):
    """Score differences decide order/cap; native targets supply references only."""
    a, b = np.array(list(combinations(range(len(test)), 2)), int).T
    p = np.array([r['p'][family] for r in test], float)
    radius = np.array([r['radius'] for r in test], float)
    truth = order(p[a]-radius[a], p[a]+radius[a], p[b]-radius[b], p[b]+radius[b])
    delta = scores[a]-scores[b]
    point = np.sign(delta).astype(int)
    return dict(a=a, b=b, truth=truth, point=point,
                budget=top_budget(point, np.abs(delta), budget))


def transitions(truth, base, new):
    resolved = truth != 0
    common = resolved & (base != 0) & (new != 0)
    return dict(common=int(common.sum()),
                repair=int(np.sum(common & (base != truth) & (new == truth))),
                damage=int(np.sum(common & (base == truth) & (new != truth))),
                both_wrong=int(np.sum(common & (base != truth) & (new != truth))),
                ties=int(np.sum(resolved & ((base == 0) | (new == 0)))))


def aggregate(cases, cfg):
    """Preserve policy-own answered sets under shared endpoint bootstrap weights."""
    codes = sorted({r['code'] for c in cases for r in c['test']})
    index = {code:i for i,code in enumerate(codes)}
    flat = {m:{p:[] for p in ('point', 'budget')} for m in METHODS}
    for case in cases:
        for q in case['queries']:
            a = np.array([index[case['test'][i]['code']] for i in q['a']])
            b = np.array([index[case['test'][i]['code']] for i in q['b']])
            for m in METHODS:
                for policy in ('point', 'budget'):
                    flat[m][policy].append((a,b,np.array(q['truth']),np.array(q['methods'][m][policy])))
    def evaluate(records, weights):
        numerator = denominator = 0.
        total = Counter()
        for a,b,truth,answer in records:
            total.update(counts(truth,answer))
            w = weights[a]*weights[b]
            valid = (truth != 0) & (answer != 0)
            numerator += np.sum(w * valid * (truth != answer))
            denominator += np.sum(w * valid)
        return rates(total), float(numerator/denominator) if denominator else None
    summaries = {m:{p:evaluate(flat[m][p], np.ones(len(codes)))[0]
                    for p in ('point','budget')} for m in METHODS}
    rng = np.random.default_rng(cfg['seed'])
    contrasts = {m:{p:[] for p in ('point','budget')} for m in METHODS if m != 'family_factor'}
    for _ in range(cfg['bootstrap_authority_repetitions']):
        weights = rng.multinomial(len(codes), np.repeat(1/len(codes),len(codes)))
        risk = {m:{p:evaluate(flat[m][p],weights)[1] for p in ('point','budget')} for m in METHODS}
        for m in contrasts:
            for policy in ('point','budget'):
                if risk[m][policy] is not None and risk['family_factor'][policy] is not None:
                    contrasts[m][policy].append(100*(risk[m][policy]-risk['family_factor'][policy]))
    intervals = {m:{p:np.quantile(v,[.025,.975]).tolist() if v else None for p,v in policies.items()}
                 for m,policies in contrasts.items()}
    transition = Counter()
    for c in cases:
        for q in c['queries']:
            transition.update(transitions(np.array(q['truth']),np.array(q['methods']['family_factor']['point']),
                                          np.array(q['methods']['pairwise_logistic']['point'])))
    return dict(authorities=len(codes), methods=summaries,
                paired_risk_difference_95_pp=intervals, ranking_factor_transitions=dict(transition))


def main():
    assert not (OUT/'summary.json').exists(), 'Preserve the recorded campaign; do not overwrite.'
    # Each run starts fresh; completed records are never overwritten.
    retained = {}
    cfg = json.loads(CFG.read_text()); base = json.loads(CONFIG.read_text())
    robust = json.loads((BASE/'summary.json').read_text())
    # Check exact retained bytes before using the parser or stored reference fits.
    for item in robust['raw_inputs']:
        assert digest(ROOT/item['path']) == item['sha256'], item['path']
    rows, inventory, provenance = load_rows(base)
    lookup = {r['path']:r['sha256'] for r in robust['cases']}
    OUT.mkdir(exist_ok=True)
    records, files = [], []
    for source in base['families']:
        for label in cfg['geographic_partitions']:
            for fold_id in range(5):
                name = f'{source}-{label}-{fold_id}'
                path = BASE/(name+'.json')
                assert digest(path) == lookup[str(path.relative_to(ROOT))]
                dest = OUT/(name+'.json')
                if dest.exists():
                    assert digest(dest) == retained[str(dest.relative_to(ROOT))]
                    result = json.loads(dest.read_text())
                    assert result['reference_case_sha256'] == digest(path)
                    files.append(dict(path=str(dest.relative_to(ROOT)),sha256=digest(dest)))
                    records.append(result)
                    print(name, 'retained converged fit', flush=True)
                    continue
                old = json.loads(path.read_text())
                train = [rows[(source,r['year'],r['code'])] for r in old['train_keys']]
                cal, test = old['calibration'], old['test']
                sets = [{r['code'] for r in rr} for rr in (train,cal,test)]
                assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
                assert all(partition(r['code'],label) not in (fold_id,(fold_id+1)%5) for r in train)
                for rr in (cal,test):
                    for row in rr:
                        raw = rows[(source,row['year'],row['code'])]
                        assert all(row[k] == raw[k] for k in row if k != 'previous')
                        if 'previous' in row:
                            assert all(row['previous'][k] == raw['previous'][k] for k in row['previous'])
                scores, fits = [], []
                start = time.perf_counter()
                for family in (0,1):
                    score, fit = fit_ranker(train,test,family,label,cfg)
                    scores.append(score); fits.append(fit)
                result = dict(name=name, source=source, partition=label,
                              reference_case_sha256=digest(path), train_keys=old['train_keys'],
                              unused_calibration_keys=[dict(code=r['code'],year=r['year']) for r in cal],
                              test=test, scores=np.column_stack(scores).tolist(), fits=fits, queries=[],
                              fit_seconds=time.perf_counter()-start)
                references = {m:queries(test,np.array(old['methods'][m]['test_prediction']),False)
                              for m in METHODS if m != 'pairwise_logistic'}
                for family in (0,1):
                    q = score_queries(test,scores[family],family,cfg['answer_budget'])
                    result_q = {k:q[k].tolist() for k in ('a','b','truth')}
                    result_q['family'] = family
                    result_q['methods'] = {'pairwise_logistic':{p:q[p].tolist() for p in ('point','budget')}}
                    for m in references:
                        r = references[m][family]
                        assert np.array_equal(r['truth'],q['truth'])
                        answer = top_budget(r['point'],r['score'],cfg['answer_budget'])
                        result_q['methods'][m] = dict(point=r['point'].tolist(),budget=answer.tolist())
                        for policy, arr in [('point',r['point']),('budget=0.75',answer)]:
                            expected = next(v for v in old['methods'][m]['metrics'] if v['family']==family and v['policy']==policy)
                            assert all(expected[k] == v for k,v in counts(r['truth'],arr).items())
                    result['queries'].append(result_q)
                dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                files.append(dict(path=str(dest.relative_to(ROOT)),sha256=digest(dest)))
                records.append(result)
                print(name, 'fits complete', flush=True)
    groups = {s+':'+p:aggregate([r for r in records if r['source']==s and r['partition']==p],cfg)
              for s in base['families'] for p in cfg['geographic_partitions']}
    summary = dict(config_sha256=digest(CFG),protocol_sha256=digest(PROTOCOL),script_sha256=digest(__file__),
                   base_config_sha256=digest(CONFIG),reference_config_sha256=digest(RCFG),
                   reference_summary_sha256=digest(BASE/'summary.json'),raw_inputs=robust['raw_inputs'],cases=files,
                   summaries=groups, selected_penalties=dict(Counter(str(f['selected_penalty']) for r in records for f in r['fits'])),
                   environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),


                   scope='Reused development sources; new prespecified reference, no fresh-source confirmation; conditional fixed-fit authority bootstrap and descriptive refitted partitions')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for group,value in groups.items():
        print(group,json.dumps(value),flush=True)


if __name__ == '__main__':
    main()
