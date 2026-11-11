"""Fixed partition, temporal, vintage, calibration and answer-budget evaluation."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import json
import math
import warnings

import numpy as np

from comparison_panel import ROOT, CONFIG, conformal_quantile, digest, fit_predict, load_rows, order
from countviews.values import parse_count
from input_readers import fold

RCFG=ROOT/'experiments/comparison/config/robustness.json'
OUT=ROOT/'results/comparison-robustness'


def archive_rows(config):
    """Use the declared, checksum-verified 2024 B1/C1 files without rewriting bytes."""
    manifest=json.loads((ROOT/'data/comparison-input-manifest.json').read_text())
    lookup={r['path']:r for r in manifest['files']}; parts=defaultdict(dict); provenance=[]
    columns=[('Anypoint_episodes','Anypoint_child'),('Referrals','Referrals_child')]
    for family,(name,cols) in enumerate(zip(config['asof_control']['raw_files'],columns)):
        assert digest(ROOT/name)==lookup[name]['sha256']
        provenance.append(dict(path=name,sha256=digest(ROOT/name)))
        with (ROOT/name).open(encoding='utf-8-sig') as stream:
            for r in csv.DictReader(stream):
                if r['geographic_level']!='Local authority':continue
                key=(int(r['time_period']),r['new_la_code'])
                assert family not in parts[key]
                parts[key][family]=(parse_count(r[cols[0]]).value,parse_count(r[cols[1]]).value,r['region_code'])
    result={}; excluded=0
    for (year,code),p in sorted(parts.items()):
        if set(p)!={0,1} or any(p[f][i] is None or p[f][i]<=0 for f in p for i in (0,1)):
            excluded+=1;continue
        assert p[0][2]==p[1][2] and p[0][2]
        result[(year,code)]=dict(source='DfE',year=year,code=code,region=p[0][2],
                                e=[p[f][0] for f in (0,1)],p=[p[f][1] for f in (0,1)],radius=0,vintage=2024)
    for (year,code),r in result.items():r['previous']=result.get((year-1,code))
    return result,provenance,excluded


def partition(code,label):
    if label=='original':return fold(code)
    return int(hashlib.sha256((label+'|'+code).encode()).hexdigest()[:8],16)%5


def top_budget(point,score,budget):
    """Select a fraction in [0,1] by unlabeled margins; preserve stable ties."""
    if not 0 <= budget <= 1:
        raise ValueError('Answer budget must be a fraction between zero and one')
    n=math.floor(budget*len(score)); mask=np.zeros(len(score),dtype=bool)
    mask[np.argsort(-score,kind='stable')[:n]]=True
    return np.where(mask,point,0)


def queries(test,pred,temporal):
    e=np.array([r['e'] for r in test],float);p=np.array([r['p'] for r in test],float)
    rad=np.array([r['radius'] for r in test])
    out=[]
    for f in (0,1):
        if temporal:
            a=np.arange(len(test)); b=np.full(len(test),-1)
            anchor=np.array([r['previous']['p'][f] for r in test])
            truth=order(p[:,f]-rad,p[:,f]+rad,anchor,anchor)
            point=order(pred[:,f],pred[:,f],anchor,anchor)
            score=np.abs(pred[:,f]-anchor)/e[:,f]
        else:
            a,b=np.array(list(combinations(range(len(test)),2)),int).T;anchor=None
            truth=order(p[a,f]-rad[a],p[a,f]+rad[a],p[b,f]-rad[b],p[b,f]+rad[b])
            point=order(pred[a,f],pred[a,f],pred[b,f],pred[b,f])
            score=np.abs(pred[a,f]-pred[b,f])/(e[a,f]+e[b,f])
        out.append(dict(f=f,a=a,b=b,truth=truth,point=point,score=score,anchor=anchor))
    return out


def counts(truth,answer):
    resolved=truth!=0;selected=answer!=0
    return dict(queries=len(truth),resolved=int(resolved.sum()),answered_all=int(selected.sum()),
                answered=int((selected&resolved).sum()),wrong=int((selected&resolved&(truth!=answer)).sum()))


def rates(c):
    c=dict(c);c['coverage']=c['answered']/c['resolved'] if c['resolved'] else None
    c['coverage_all']=c['answered_all']/c['queries'] if c['queries'] else None
    c['risk']=c['wrong']/c['answered'] if c['answered'] else None
    return c


def serial_row(r):
    v={k:r[k] for k in ('source','year','code','region','e','p','radius')}
    if r.get('previous'):v['previous']={k:r['previous'][k] for k in ('year','e','p')}
    return v


def run_case(case,base,cfg):
    train,cal,test=case['train'],case['cal'],case['test'];temporal=case['task']!='geographic'
    assert min(map(len,(train,cal,test)))>=2
    if not temporal:
        sets=[{r['code'] for r in rr} for rr in (train,cal,test)]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    else:
        assert max(r['year'] for r in train)<min(r['year'] for r in cal)<min(r['year'] for r in test)
    result={k:case[k] for k in ('name','source','task','group')};result.update(
        train_keys=[dict(code=r['code'],year=r['year']) for r in train],
        calibration=[serial_row(r) for r in cal],test=[serial_row(r) for r in test],methods={})
    methods=base['models']+(['lagged_factor'] if temporal else [])
    for method in methods:
        with warnings.catch_warnings(record=True) as ww:
            warnings.simplefilter('always')
            pp=np.column_stack([fit_predict(method,train,cal+test,f,base,temporal) for f in (0,1)])
        assert np.isfinite(pp).all()
        cp,tp=pp[:len(cal)],pp[len(cal):]
        ep=np.array([r['e'] for r in cal]);native=np.array([r['p'] for r in cal]);rad=np.array([r['radius'] for r in cal])[:,None]
        scores=np.max((np.abs(native-cp)+rad)/ep,axis=1)
        ee=np.array([r['e'] for r in test]);tt=np.array([r['p'] for r in test]);rr=np.array([r['radius'] for r in test])[:,None]
        qq=queries(test,tp,temporal);metrics=[]
        for q in qq:
            f=q['f'];fit=dict(mape_sum=float(np.sum(np.abs(tp[:,f]-tt[:,f])/tt[:,f])*100),
                              absolute_error_sum=float(np.sum(np.abs(tp[:,f]-tt[:,f]))),native_sum=float(tt[:,f].sum()),rows=len(test))
            metrics.append(dict(family=f,policy='point',**counts(q['truth'],q['point']),**fit))
        quantiles={}
        for alpha in cfg['alpha_values']:
            quantile=conformal_quantile(scores,alpha);quantiles[str(alpha)]=quantile if math.isfinite(quantile) else None
            low=np.maximum(0,tp-quantile*ee);high=tp+quantile*ee
            cover=(low<=np.maximum(0,tt-rr))&(high>=tt+rr)
            for q in qq:
                f=q['f'];a=q['a'];b=q['b']
                answer=order(low[:,f],high[:,f],q['anchor'],q['anchor']) if temporal else order(low[a,f],high[a,f],low[b,f],high[b,f])
                metrics.append(dict(family=f,policy='alpha='+str(alpha),**counts(q['truth'],answer),rows=len(test),
                                    covered_rows=int(cover[:,f].sum()),covered_vectors=int(cover.all(axis=1).sum())))
        for budget in cfg['answer_budgets']:
            for q in qq:
                answer=top_budget(q['point'],q['score'],budget)
                metrics.append(dict(family=q['f'],policy='budget='+str(budget),**counts(q['truth'],answer)))
        result['methods'][method]=dict(calibration_prediction=cp.tolist(),test_prediction=tp.tolist(),
                                      calibration_scores=scores.tolist(),quantiles=quantiles,metrics=metrics,
                                      numerical_warnings=dict(count=len(ww),messages=sorted({str(w.message) for w in ww})))
    return result


def aggregate(results):
    totals=defaultdict(lambda:defaultdict(float))
    for r in results:
        for method,m in r['methods'].items():
            for v in m['metrics']:
                for family in ('all',str(v['family'])):
                    key=(r['group'],method,v['policy'],family)
                    for k,x in v.items():
                        if k not in ('family','policy'):totals[key][k]+=x
    answer=[]
    for (group,method,policy,family),s in sorted(totals.items()):
        t=rates(s)
        if policy=='point':t.update(mape=t['mape_sum']/t['rows'],wape=100*t['absolute_error_sum']/t['native_sum'])
        if policy.startswith('alpha='):t.update(row_bin_coverage=t['covered_rows']/t['rows'],vector_coverage=t['covered_vectors']/t['rows'])
        answer.append(dict(group=group,method=method,policy=policy,family=family,**t))
    return answer


def main():
    base=json.loads(CONFIG.read_text());cfg=json.loads(RCFG.read_text());OUT.mkdir(exist_ok=True)
    rows,inventory,provenance=load_rows(base);arch,arch_prov,arch_excluded=archive_rows(cfg)
    cases=[]
    for source in base['families']:
        available=[r for (s,y,_),r in rows.items() if s==source and y==2025]
        for part in cfg['geographic_partitions']:
            for j in range(5):
                choose=lambda r:partition(r['code'],part)
                cases.append(dict(name=f'{source}-{part}-{j}',source=source,task='geographic',group=source+':'+part,
                                  train=[r for r in available if choose(r) not in (j,(j+1)%5)],
                                  cal=[r for r in available if choose(r)==(j+1)%5],test=[r for r in available if choose(r)==j]))
    hist=[r for (s,_,_),r in rows.items() if s=='DfE' and r['previous'] is not None]
    for year in cfg['temporal_origins']:
        cases.append(dict(name=f'DfE-current-{year}',source='DfE',task='temporal',group=f'DfE:current:{year}',
                          train=[r for r in hist if 2014<=r['year']<=year-2],cal=[r for r in hist if r['year']==year-1],test=[r for r in hist if r['year']==year]))
    common=sorted(code for (s,y,code) in rows if s=='DfE' and y==2025 and ('DfE',2024,code) in rows and (2024,code) in arch)
    older=[r for r in arch.values() if r['previous'] is not None]
    test=[]
    for code in common:
        r=dict(rows[('DfE',2025,code)]);r['previous']=arch[(2024,code)];test.append(r)
    cases.append(dict(name='DfE-preceding-release-2025',source='DfE',task='asof',group='DfE:preceding:2025',
                      train=[r for r in older if 2014<=r['year']<=2023],cal=[r for r in older if r['year']==2024],test=test))
    cases.append(dict(name='DfE-matched-current-2025',source='DfE',task='temporal',group='DfE:matched-current:2025',
                      train=[r for r in hist if 2014<=r['year']<=2023],cal=[r for r in hist if r['year']==2024],test=[rows[('DfE',2025,c)] for c in common]))
    results=[];files=[]
    for case in cases:
        r=run_case(case,base,cfg);p=OUT/(case['name']+'.json')
        p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');results.append(r);files.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p)))
        print(case['name'],len(case['train']),len(case['cal']),len(case['test']),flush=True)
    changes=[]
    for code in common:
        r=rows[('DfE',2025,code)]
        for f in (0,1):
            new=rows[('DfE',2024,code)]['p'][f];old=arch[(2024,code)]['p'][f];target=r['p'][f]
            changes.append(dict(code=code,family=f,anchor_current=new,anchor_preceding=old,target=target,
                                direction_changed=int(np.sign(target-new))!=int(np.sign(target-old))))
    result=dict(config_sha256=digest(RCFG),script_sha256=digest(__file__),base_config_sha256=digest(CONFIG),
                raw_inputs=provenance+arch_prov,archived_excluded_vectors=arch_excluded,cases=files,
                summaries=aggregate(results),vintage_anchor_changes=changes,
                dependence='Same authority populations, targets, folds and years are reused; split/origin ranges are descriptive, not independent replications.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
