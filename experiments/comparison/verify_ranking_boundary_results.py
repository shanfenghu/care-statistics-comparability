"""Numerical and policy checks for verify ranking boundary results."""
from collections import Counter
from itertools import combinations
import json
import math
import numpy as np
from comparison_panel import ROOT,digest,features
from comparison_ranking import OUT as OLD_OUT
from comparison_ranking_boundary import OUT,CFG,PROTOCOL


def main():
    path=OUT/'summary.json';summary=json.loads(path.read_text());cfg=json.loads(CFG.read_text())
    assert summary['script_sha256']==digest(ROOT/'experiments/comparison/comparison_ranking_boundary.py')
    assert summary['config_sha256']==digest(CFG) and summary['protocol_sha256']==digest(PROTOCOL)
    manifest=json.loads((OUT/'selection-manifest.json').read_text())
    assert summary['selection_manifest_sha256']==digest(OUT/'selection-manifest.json')
    selection_check=json.loads((OUT/'selection-verification.json').read_text())
    assert selection_check['selection_manifest_sha256']==summary['selection_manifest_sha256']
    assert selection_check['verifier_sha256']==digest(ROOT/'experiments/comparison/verify_ranking_boundary_selection.py')
    selected={}
    for item in manifest['models']:
        assert digest(ROOT/item['path'])==item['sha256']
        r=json.loads((ROOT/item['path']).read_text());selected[(r['name'],r['family'])]=r
    totals={};families={};max_score_error=0.;replays=0;extra={};changes=Counter()
    for item in summary['cases']:
        assert digest(ROOT/item['path'])==item['sha256']
        c=json.loads((ROOT/item['path']).read_text());old=json.loads((OLD_OUT/(c['name']+'.json')).read_text())
        assert digest(OLD_OUT/(c['name']+'.json'))==c['original_ranking_case_sha256']
        assert c['test']==old['test'] and c['train_keys']==old['train_keys'] and c['unused_calibration_keys']==old['unused_calibration_keys']
        group=c['source']+':'+c['partition'];test=c['test']
        for f,fit in enumerate(c['fits']):
            assert fit==selected[(c['name'],f)]['fit']
            for i,r in enumerate(test):
                x=features(r,f)
                score=math.fsum((x.get(k,0.)-m)/s*w for k,m,s,w in zip(fit['feature_names'],fit['mean'],fit['scale'],fit['coefficients']))
                max_score_error=max(max_score_error,abs(score-c['scores'][i][f]))
        for q,oq in zip(c['queries'],old['queries']):
            f=q['family'];truth=[];point=[];margin=[]
            for i,j in zip(q['a'],q['b']):
                gap=test[i]['p'][f]-test[j]['p'][f];r=test[i]['radius']+test[j]['radius']
                truth.append(1 if gap>r else -1 if gap < -r else 0)
                d=c['scores'][i][f]-c['scores'][j][f];point.append(1 if d>0 else -1 if d<0 else 0);margin.append(abs(d))
            assert truth==q['truth'] and q['methods']['pairwise_logistic']['point']==point
            chosen=set(sorted(range(len(margin)),key=lambda i:(-margin[i],i))[:math.floor(cfg['answer_budget']*len(margin))])
            budget=[d if i in chosen else 0 for i,d in enumerate(point)]
            assert budget==q['methods']['pairwise_logistic']['budget']
            assert q['methods']['pairwise_original']==oq['methods']['pairwise_logistic']
            for method in ['family_factor','context_ridge','context_forest']:assert q['methods'][method]==oq['methods'][method]
            for method,policies in q['methods'].items():
                for policy,answer in policies.items():
                    total=totals.setdefault((group,method,policy),Counter());family=families.setdefault((group,method,policy,f),Counter())
                    for t,a in zip(truth,answer):
                        value=dict(queries=1,resolved=int(t!=0),answered_all=int(a!=0),answered=int(t!=0 and a!=0),wrong=int(t!=0 and a!=0 and t!=a))
                        total.update(value);family.update(value)
                    replays+=1
            base=q['methods']['family_factor']['point'];oldpoint=q['methods']['pairwise_original']['point']
            e=extra.setdefault(group,Counter())
            for t,b,n,o in zip(truth,base,point,oldpoint):
                if t and not b and n:e.update(answers=1,wrong=int(t!=n))
                if t and n!=o:changes[group]+=1
    for (group,method,policy),count in totals.items():
        saved=(summary['original_reference_summaries'][group]['methods']['pairwise_logistic'][policy]
               if method=='pairwise_original' else summary['summaries'][group]['methods'][method][policy])
        assert all(saved[k]==v for k,v in count.items())
    assert max_score_error<1e-9
    result=dict(summary_sha256=digest(path),verifier_sha256=digest(__file__),
                selection_verification_sha256=digest(OUT/'selection-verification.json'),
                preserved_selection_manifest_sha256=summary['selection_manifest_sha256'],
                cases=30,scalar_policy_family_replays=replays,max_abs_scalar_score_error=max_score_error,
                all_original_reference_answers_and_splits_exact=True,all_selected_fits_equal_preserved_choices=True,
                all_point_and_budget_counts_match=True,
                factor_tie_additional_rank_answers={g:dict(v) for g,v in extra.items()},
                resolved_point_directions_changed_from_original=dict(changes),
                family_counts=[dict(group=k[0],method=k[1],policy=k[2],family=k[3],**v) for k,v in sorted(families.items())],
                scope='Replay/label-isolation validation, not independent scientific replication or external validity')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='family_counts'},indent=2))


if __name__=='__main__':main()
