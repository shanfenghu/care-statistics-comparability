"""Numerical and policy checks for verify comparison robustness."""
from itertools import combinations
import json
import math

from comparison_panel import CONFIG, ROOT, digest, load_rows
from comparison_robustness import RCFG, OUT, archive_rows
from verify_comparison_panel import scalar_ridge, interval_order


def main():
    base=json.loads(CONFIG.read_text());cfg=json.loads(RCFG.read_text());s=json.loads((OUT/'summary.json').read_text())
    rows,_,_=load_rows(base);older,_,_=archive_rows(cfg)
    assert s['config_sha256']==digest(RCFG)
    assert s['script_sha256']==digest(ROOT/'experiments/comparison/comparison_robustness.py')
    fits=metrics=0;maximum=0.0;all_versions=[]
    for (year,code),r in older.items():
        cur=rows.get(('DfE',year,code))
        if cur:all_versions.append(r['e']==cur['e'] and r['p']==cur['p'] and r['region']==cur['region'])
    for info in s['cases']:
        assert digest(ROOT/info['path'])==info['sha256'];c=json.loads((ROOT/info['path']).read_text())
        temporal=c['task']!='geographic';asof=c['task']=='asof'
        train=[older[(r['year'],r['code'])] if asof else rows[(c['source'],r['year'],r['code'])] for r in c['train_keys']]
        evaluate=c['calibration']+c['test']
        for f in (0,1):
            expected=scalar_ridge(train,evaluate,f,base['ridge_alpha'],temporal)
            got=[r[f] for r in c['methods']['context_ridge']['calibration_prediction']+c['methods']['context_ridge']['test_prediction']]
            for a,b in zip(expected,got):
                maximum=max(maximum,abs(a-b));assert abs(a-b)<=1e-8*max(1,abs(b))
            fits+=1
        for method,data in c['methods'].items():
            pred=data['test_prediction'];test=c['test']
            for f in (0,1):
                pairs=[(i,None) for i in range(len(test))] if temporal else list(combinations(range(len(test)),2))
                truth=[];point=[];score=[]
                for i,j in pairs:
                    a=test[i];b=test[j] if j is not None else None;mu=pred[i][f]
                    lo=b['p'][f]-b['radius'] if b else a['previous']['p'][f];hi=b['p'][f]+b['radius'] if b else lo
                    truth.append(interval_order(a['p'][f]-a['radius'],a['p'][f]+a['radius'],lo,hi))
                    other=pred[j][f] if b else lo;point.append(interval_order(mu,mu,other,other))
                    score.append(abs(mu-other)/(a['e'][f]+b['e'][f] if b else a['e'][f]))
                for record in [v for v in data['metrics'] if v['family']==f]:
                    policy=record['policy'];answer=list(point)
                    if policy.startswith('budget='):
                        k=math.floor(float(policy.split('=')[1])*len(pairs))
                        selected=set(sorted(range(len(score)),key=lambda i:(-score[i],i))[:k])
                        answer=[v if i in selected else 0 for i,v in enumerate(point)]
                        assert sum(bool(v) for v in answer)<=k
                    elif policy.startswith('alpha='):
                        q=data['quantiles'][policy.split('=')[1]];q=math.inf if q is None else q
                        answer=[]
                        for i,j in pairs:
                            a=test[i];lo=max(0,pred[i][f]-q*a['e'][f]);hi=pred[i][f]+q*a['e'][f]
                            if j is None:other_lo=other_hi=a['previous']['p'][f]
                            else:
                                b=test[j];other_lo=max(0,pred[j][f]-q*b['e'][f]);other_hi=pred[j][f]+q*b['e'][f]
                            answer.append(interval_order(lo,hi,other_lo,other_hi))
                    assert record['resolved']==sum(bool(t) for t in truth)
                    assert record['answered_all']==sum(bool(a) for a in answer)
                    assert record['answered']==sum(bool(t and a) for t,a in zip(truth,answer))
                    assert record['wrong']==sum(bool(t and a and t!=a) for t,a in zip(truth,answer))
                    metrics+=1
    # Check the original partition really reproduces the prior fixed panel.
    for source in ('DHSC','DfE'):
        for j in range(5):
            c=json.loads((OUT/f'{source}-original-{j}.json').read_text())
            old=json.loads((ROOT/f'results/comparison-panel/{source}-geographic-{j}.json').read_text())
            for method in base['models']:
                expected={(r['code'],r['family']):r['prediction'] for r in old['predictions'][method]}
                for row,pp in zip(c['test'],c['methods'][method]['test_prediction']):
                    for f in (0,1):assert abs(expected[(row['code'],f)]-pp[f])<1e-9
    # A null vintage claim includes every prediction and policy, not just counts.
    preceding=json.loads((OUT/'DfE-preceding-release-2025.json').read_text())
    matched=json.loads((OUT/'DfE-matched-current-2025.json').read_text())
    assert preceding['test']==matched['test']
    assert preceding['calibration']==matched['calibration']
    for method in preceding['methods']:
        assert preceding['methods'][method]==matched['methods'][method]
    result=dict(cases=len(s['cases']),scalar_ridge_fits=fits,max_absolute_prediction_difference=maximum,
                scalar_policy_family_checks=metrics,original_partition_reproduced=True,matched_vintage_predictions_and_policies_identical=True,
                overlapping_2024_2025_authority_year_vectors=len(all_versions),identical_selected_count_and_region_vectors=sum(all_versions),
                native_anchor_direction_changes=sum(r['direction_changed'] for r in s['vintage_anchor_changes']),
                summary_sha256=digest(OUT/'summary.json'),verifier_sha256=digest(__file__))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
