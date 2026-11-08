"""Numerical and policy checks for verify comparison ranking."""
from collections import Counter
from itertools import combinations
import json
import math

import numpy as np

from comparison_panel import ROOT, CONFIG, digest, features, load_rows
from comparison_ranking import CFG, OUT, PROTOCOL, METHODS
from comparison_robustness import partition


def main():
    cfg=json.loads(CFG.read_text());summary=json.loads((OUT/'summary.json').read_text())
    assert summary['script_sha256']==digest(ROOT/'experiments/comparison/comparison_ranking.py')
    assert summary['config_sha256']==digest(CFG) and summary['protocol_sha256']==digest(PROTOCOL)
    rows,_,_=load_rows(json.loads(CONFIG.read_text()))
    totals={};family_totals={};score_error=0.;gradient_norm=0.;replayed=0;fits=0;tie_extras={}
    for item in summary['cases']:
        path=ROOT/item['path'];assert digest(path)==item['sha256']
        case=json.loads(path.read_text());test=case['test'];source=case['source'];label=case['partition']
        train=[rows[(source,k['year'],k['code'])] for k in case['train_keys']]
        train_codes={r['code'] for r in train};test_codes={r['code'] for r in test}
        cal_codes={r['code'] for r in case['unused_calibration_keys']}
        assert not (train_codes&test_codes or train_codes&cal_codes or test_codes&cal_codes)
        for family,fit in enumerate(case['fits']):
            assert fit['selected_penalty']==min(fit['inner_candidates'],key=lambda c:(c['loss'],-c['penalty']))['penalty']
            def transform(row):
                values=features(row,family)
                return [(values.get(name,0.)-mean)/scale for name,mean,scale in zip(fit['feature_names'],fit['mean'],fit['scale'])]
            w=fit['coefficients'];x=[transform(r) for r in train]
            gradients=[[] for _ in w]
            for i,j in combinations(range(len(train)),2):
                gap=train[i]['p'][family]-train[j]['p'][family]
                if abs(gap)<=train[i]['radius']+train[j]['radius']:continue
                y=1 if gap>0 else -1
                difference=[a-b for a,b in zip(x[i],x[j])]
                margin=y*math.fsum(a*b for a,b in zip(difference,w))
                weight=math.exp(-margin)/(1+math.exp(-margin)) if margin>=0 else 1/(1+math.exp(margin))
                for k,value in enumerate(difference):gradients[k].append(-y*value*weight)
            assert len(gradients[0])==fit['training_pairs']
            grad=[math.fsum(v)/len(v)+fit['selected_penalty']*c for v,c in zip(gradients,w)]
            gradient_norm=max(gradient_norm,max(abs(v) for v in grad));fits+=1
            for i,row in enumerate(test):
                value=math.fsum(a*b for a,b in zip(transform(row),w))
                score_error=max(score_error,abs(value-case['scores'][i][family]))
        for q in case['queries']:
            family=q['family'];a=q['a'];b=q['b'];truth=[];point=[];margins=[]
            for i,j in zip(a,b):
                gap=test[i]['p'][family]-test[j]['p'][family]
                radius=test[i]['radius']+test[j]['radius']
                truth.append(1 if gap>radius else -1 if gap < -radius else 0)
                delta=case['scores'][i][family]-case['scores'][j][family]
                point.append(1 if delta>0 else -1 if delta<0 else 0);margins.append(abs(delta))
            assert truth==q['truth'] and point==q['methods']['pairwise_logistic']['point']
            selected=set(sorted(range(len(margins)),key=lambda i:(-margins[i],i))[:math.floor(cfg['answer_budget']*len(margins))])
            budget=[value if i in selected else 0 for i,value in enumerate(point)]
            assert budget==q['methods']['pairwise_logistic']['budget']
            group=source+':'+label
            for method in METHODS:
                for policy in ('point','budget'):
                    answer=q['methods'][method][policy];key=(group,method,policy)
                    total=totals.setdefault(key,Counter())
                    family_total=family_totals.setdefault((group,method,policy,family),Counter())
                    for t,v in zip(truth,answer):
                        value=dict(queries=1,resolved=int(t!=0),answered_all=int(v!=0),
                                   answered=int(t!=0 and v!=0),wrong=int(t!=0 and v!=0 and t!=v))
                        total.update(value);family_total.update(value)
                    replayed+=1
            base=q['methods']['family_factor']['point']
            extra=tie_extras.setdefault(group,Counter())
            for t,bv,nv in zip(truth,base,point):
                if t and not bv and nv:extra.update(answers=1,wrong=int(t!=nv))
    for (group,method,policy),total in totals.items():
        saved=summary['summaries'][group]['methods'][method][policy]
        assert all(saved[k]==v for k,v in total.items())
    assert score_error<1e-10 and gradient_norm<1e-5
    record=dict(summary_sha256=digest(OUT/'summary.json'),verifier_sha256=digest(__file__),cases=30,
                final_family_fits= fits,scalar_policy_family_replays=replayed,
                max_abs_scalar_score_error=score_error,max_scalar_gradient_inf=gradient_norm,
                all_inner_selection_choices_match_recorded_train_only_losses=True,
                all_authority_splits_disjoint=True,all_new_point_and_budget_counts_match_scalar_replay=True,
                factor_tie_additional_rank_answers={k:dict(v) for k,v in tie_extras.items()},
                family_counts=[dict(group=k[0],method=k[1],policy=k[2],family=k[3],**v)
                               for k,v in sorted(family_totals.items())],
                family_count_scope='Post-outcome disaggregation of every family/partition; no refitting or selection',
                scope='Numerical/label-isolation checks, not external validity or independent scientific replication')
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
