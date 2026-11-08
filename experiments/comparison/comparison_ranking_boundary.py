"""Training-only extension of a fixed ranking grid before outer scoring."""
from collections import Counter
from copy import deepcopy
import argparse
import json

import numpy as np

from comparison_panel import ROOT, CONFIG, digest, load_rows
from comparison_robustness import partition, top_budget
from comparison_ranking import (CFG as OLD_CFG, OUT as OLD_OUT, prepare, pair_data,
                                fit_pairs, score_queries, aggregate)

CFG=ROOT/'experiments/comparison/config/ranking-expanded.json'
PROTOCOL=ROOT/'experiments/comparison/config/ranking-expanded-method.txt'
OUT=ROOT/'results/comparison-ranking-boundary'


def extend_selection(train, family, label, old_fit, cfg):
    """Extend cached training-only validation losses without outer observations."""
    inner=[];split_records=[]
    for fold_id in sorted({partition(r['code'],label) for r in train}):
        fitting=[r for r in train if partition(r['code'],label)!=fold_id]
        validating=[r for r in train if partition(r['code'],label)==fold_id]
        assert not ({r['code'] for r in fitting}&{r['code'] for r in validating})
        x,xx,vector,scale=prepare(fitting,validating,family)
        difference,direction=pair_data(fitting,x,family)
        vd,vy=pair_data(validating,xx,family)
        inner.append((fold_id,difference,direction,vd,vy))
        split_records.append(dict(fold=fold_id,training_codes=[r['code'] for r in fitting],
                                  validation_codes=[r['code'] for r in validating],
                                  feature_names=vector.get_feature_names_out().tolist(),
                                  mean=scale.mean_.tolist(),scale=scale.scale_.tolist()))
    assert len(inner)==3
    candidates=deepcopy(old_fit['inner_candidates']);validation_directions={}
    for penalty in cfg['new_regularization']:
        wrong=0.;resolved=0;checks=[]
        for fold_id,difference,direction,vd,vy in inner:
            w,status=fit_pairs(difference,direction,penalty,cfg)
            answer=np.sign(np.einsum('ij,j->i',vd,w,optimize=False))
            loss=float(np.sum((answer!=vy)&(answer!=0))+.5*np.sum(answer==0))
            wrong+=loss;resolved+=len(vy)
            validation_directions[(penalty,fold_id)]=answer
            checks.append(dict(fold=fold_id,resolved=len(vy),wrong_plus_half_ties=loss,
                               training_pairs=len(direction),coefficients=w.tolist(),optimizer=status))
        candidates.append(dict(penalty=penalty,loss=wrong/resolved,resolved=resolved,
                               wrong_plus_half_ties=wrong,folds=checks))
    selected=min(candidates,key=lambda c:(c['loss'],-c['penalty']))['penalty']
    if selected==old_fit['selected_penalty']:
        fit=deepcopy(old_fit);reused=True
    else:
        # An old-grid value cannot newly beat the old winner under the same rule.
        assert selected in cfg['new_regularization']
        x,_,vector,scale=prepare(train,train,family)
        difference,direction=pair_data(train,x,family)
        w,status=fit_pairs(difference,direction,selected,cfg)
        fit=dict(selected_penalty=selected,training_pairs=len(direction),
                 feature_names=vector.get_feature_names_out().tolist(),
                 mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),
                 coefficients=w.tolist(),optimizer=status);reused=False
    fit['inner_candidates']=candidates
    weakest=sorted(cfg['new_regularization'])[:2]
    disagreements=sum(int(np.sum(validation_directions[(weakest[0],fold)]!=
                                    validation_directions[(weakest[1],fold)])) for fold,*_ in inner)
    return dict(fit=fit,inner_splits=split_records,reused_original_final_fit=reused,
                original_penalty=old_fit['selected_penalty'],
                original_best_inner_loss=min(c['loss'] for c in old_fit['inner_candidates']),
                expanded_best_inner_loss=min(c['loss'] for c in candidates),
                weakest_validation_disagreements=disagreements,
                weakest_validation_pairs=sum(len(t[-1]) for t in inner))


def dependencies():
    """Pin shared fitting/feature code; do not silently reinterpret cached losses."""
    files=['experiments/comparison/comparison_ranking.py','experiments/comparison/comparison_panel.py',
           'experiments/comparison/comparison_robustness.py']
    return {p:digest(ROOT/p) for p in files}


def load_inputs():
    cfg=json.loads(CFG.read_text());old=json.loads((OLD_OUT/'summary.json').read_text())
    assert digest(ROOT/'experiments/comparison/comparison_ranking.py')==old['script_sha256']
    assert digest(OLD_CFG)==old['config_sha256']
    assert not (set(cfg['new_regularization']) & set(json.loads(OLD_CFG.read_text())['regularization']))
    for r in old['raw_inputs']:assert digest(ROOT/r['path'])==r['sha256']
    for r in old['cases']:assert digest(ROOT/r['path'])==r['sha256']
    return cfg,old


def select():
    cfg,old=load_inputs();rows,_,_=load_rows(json.loads(CONFIG.read_text()))
    folder=OUT/'selections';folder.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'selection-manifest.json').exists(),'Preserve completed selection.'
    files=[];choices=[]
    for item in old['cases']:
        case=json.loads((ROOT/item['path']).read_text())
        train=[rows[(case['source'],r['year'],r['code'])] for r in case['train_keys']]
        for family in (0,1):
            path=folder/(case['name']+f'-family-{family}.json')
            if path.exists():
                record=json.loads(path.read_text())
                assert record['config_sha256']==digest(CFG) and record['script_sha256']==digest(__file__)
                assert record['reference_case_sha256']==item['sha256']
            else:
                record=extend_selection(train,family,case['partition'],case['fits'][family],cfg)
                record.update(name=case['name'],source=case['source'],partition=case['partition'],family=family,
                              reference_case_path=item['path'],reference_case_sha256=item['sha256'],
                              config_sha256=digest(CFG),protocol_sha256=digest(PROTOCOL),
                              script_sha256=digest(__file__),dependency_sha256=dependencies())
                path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            files.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path)))
            choices.append(record)
            print(case['name'],family,'training selection saved',flush=True)
    manifest=dict(stage='All training-only choices frozen; no new outer outcomes scored',
                  config_sha256=digest(CFG),protocol_sha256=digest(PROTOCOL),script_sha256=digest(__file__),
                  dependency_sha256=dependencies(),original_summary_sha256=digest(OLD_OUT/'summary.json'),
                  models=files,
                  selected_penalties=dict(Counter(str(c['fit']['selected_penalty']) for c in choices)),
                  changed_choices=sum(not c['reused_original_final_fit'] for c in choices),
                  weakest_validation_disagreements=sum(c['weakest_validation_disagreements'] for c in choices),
                  weakest_validation_pairs=sum(c['weakest_validation_pairs'] for c in choices),
                  smallest_grid_selections=sum(c['fit']['selected_penalty']==min(cfg['new_regularization']) for c in choices))
    path=OUT/'selection-manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Selection manifest',digest(path),flush=True)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['models','dependency_sha256']},indent=2),flush=True)


def score_rows(rows,fit,family):
    """Use only permitted event/region features; labels are not score inputs."""
    from comparison_panel import features
    x=np.array([[(features(r,family).get(name,0.)-mean)/scale
                 for name,mean,scale in zip(fit['feature_names'],fit['mean'],fit['scale'])] for r in rows])
    return np.einsum('ij,j->i',x,np.array(fit['coefficients']),optimize=False)


def old_new_contrasts(cases,cfg):
    """Conditional paired own-answer-set risk differences, expanded minus old."""
    codes=sorted({r['code'] for c in cases for r in c['test']});index={s:i for i,s in enumerate(codes)}
    a=[];b=[];truth=[];answers={m:{p:[] for p in ['point','budget']} for m in ['pairwise_logistic','pairwise_original']}
    for c in cases:
        for q in c['queries']:
            a.extend(index[c['test'][i]['code']] for i in q['a']);b.extend(index[c['test'][i]['code']] for i in q['b'])
            truth.extend(q['truth'])
            for m in answers:
                for p in answers[m]:answers[m][p].extend(q['methods'][m][p])
    a=np.array(a);b=np.array(b);truth=np.array(truth);resolved=truth!=0
    answers={m:{p:np.array(v) for p,v in vv.items()} for m,vv in answers.items()}
    rng=np.random.default_rng(cfg['seed']);samples={p:[] for p in ['point','budget']}
    for _ in range(cfg['bootstrap_authority_repetitions']):
        weights=rng.multinomial(len(codes),np.repeat(1/len(codes),len(codes)));w=weights[a]*weights[b]
        for p in samples:
            risks=[]
            for m in answers:
                v=answers[m][p];valid=resolved&(v!=0);den=np.sum(w*valid)
                risks.append(np.sum(w*valid*(v!=truth))/den if den else None)
            if all(r is not None for r in risks):samples[p].append(100*(risks[0]-risks[1]))
    return {p:np.quantile(v,[.025,.975]).tolist() if v else None for p,v in samples.items()}


def evaluate(expected):
    cfg,old=load_inputs();manifest_path=OUT/'selection-manifest.json'
    assert digest(manifest_path)==expected,'Selection must be fixed before outer evaluation.'
    manifest=json.loads(manifest_path.read_text())
    assert manifest['script_sha256']==digest(__file__) and manifest['config_sha256']==digest(CFG)
    selections={}
    for item in manifest['models']:
        assert digest(ROOT/item['path'])==item['sha256']
        r=json.loads((ROOT/item['path']).read_text());selections[(r['name'],r['family'])]=r
    folder=OUT/'cases';folder.mkdir(exist_ok=True)
    assert not (OUT/'summary.json').exists(),'Preserve the recorded outcome summary.'
    cases=[];files=[]
    for item in old['cases']:
        original=json.loads((ROOT/item['path']).read_text());case=deepcopy(original)
        case['fits']=[selections[(case['name'],f)]['fit'] for f in (0,1)]
        scores=[score_rows(case['test'],case['fits'][f],f) for f in (0,1)]
        case['scores']=np.column_stack(scores).tolist();case['original_ranking_case_sha256']=item['sha256']
        for q in case['queries']:
            f=q['family'];r=score_queries(case['test'],scores[f],f,cfg['answer_budget'])
            assert r['truth'].tolist()==q['truth']
            q['methods']['pairwise_original']=deepcopy(q['methods']['pairwise_logistic'])
            q['methods']['pairwise_logistic']={p:r[p].tolist() for p in ['point','budget']}
        path=folder/(case['name']+'.json');path.write_text(json.dumps(case,indent=2,allow_nan=False)+'\n')
        files.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path)));cases.append(case)
    groups=sorted({r['source']+':'+r['partition'] for r in cases})
    grouped={g:[r for r in cases if r['source']+':'+r['partition']==g] for g in groups}
    summary=dict(config_sha256=digest(CFG),protocol_sha256=digest(PROTOCOL),script_sha256=digest(__file__),
                 dependency_sha256=dependencies(),selection_manifest_sha256=expected,
                 original_summary_sha256=digest(OLD_OUT/'summary.json'),raw_inputs=old['raw_inputs'],cases=files,
                 summaries={g:aggregate(rr,cfg) for g,rr in grouped.items()},
                 expanded_minus_original_risk_95_pp={g:old_new_contrasts(rr,cfg) for g,rr in grouped.items()},
                 original_reference_summaries=old['summaries'],
                 scope='Data-informed development follow-up; choices fixed before new outer scoring; fixed-fit conditional bootstrap, dependent partitions')
    path=OUT/'summary.json';path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('All expanded outcomes saved',digest(path),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['select','evaluate'])
    parser.add_argument('--expected-selections')
    args=parser.parse_args()
    if args.stage=='select':select()
    else:
        assert args.expected_selections,'Supply the preserved selection manifest SHA-256.'
        evaluate(args.expected_selections)
