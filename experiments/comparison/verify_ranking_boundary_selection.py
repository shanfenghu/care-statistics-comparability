"""Numerical and policy checks for verify ranking boundary selection."""
from itertools import combinations
import json
import numpy as np
from comparison_panel import ROOT,CONFIG,digest,features,load_rows
from comparison_robustness import partition
from comparison_ranking_boundary import CFG,PROTOCOL,OUT


def matrix(rows,family,fit):
    """Reconstruct saved training transforms by explicit column operations."""
    return np.array([[(features(r,family).get(k,0.)-m)/s
                      for k,m,s in zip(fit['feature_names'],fit['mean'],fit['scale'])] for r in rows])


def pairs(rows,x,family):
    """Build resolved directions directly from released centers and radii."""
    difference=[];truth=[]
    for i,j in combinations(range(len(rows)),2):
        gap=rows[i]['p'][family]-rows[j]['p'][family]
        if abs(gap)>rows[i]['radius']+rows[j]['radius']:
            difference.append(x[i]-x[j]);truth.append(1 if gap>0 else -1)
    return np.array(difference),np.array(truth)


def check_fit(d,y,w,penalty,status):
    """Independent sum-based loss/gradient, not the fitted contraction code."""
    w=np.array(w);margin=y*np.sum(d*w,axis=1)
    e=np.exp(-np.abs(margin));prob=np.where(margin>=0,e/(1+e),1/(1+e))
    gradient=np.sum(d*(-y*prob)[:,None],axis=0)/len(y)+penalty*w
    loss=float(np.mean(np.logaddexp(0,-margin))+penalty*np.sum(w*w)/2)
    assert status['success'] and np.isfinite(w).all()
    assert abs(loss-status['objective'])<1e-10
    norm=float(np.max(np.abs(gradient)));assert norm<1e-5
    return norm,float(np.linalg.norm(w))


def main():
    cfg=json.loads(CFG.read_text());path=OUT/'selection-manifest.json';manifest=json.loads(path.read_text())
    assert manifest['config_sha256']==digest(CFG) and manifest['protocol_sha256']==digest(PROTOCOL)
    assert manifest['script_sha256']==digest(ROOT/'experiments/comparison/comparison_ranking_boundary.py')
    rows,_,_=load_rows(json.loads(CONFIG.read_text()));inner_checks=0;final_checks=0;max_grad=0.;max_norm=0.;loss_gains=[]
    for item in manifest['models']:
        assert digest(ROOT/item['path'])==item['sha256']
        record=json.loads((ROOT/item['path']).read_text());family=record['family']
        assert digest(ROOT/record['reference_case_path'])==record['reference_case_sha256']
        old=json.loads((ROOT/record['reference_case_path']).read_text())
        train=[rows[(record['source'],k['year'],k['code'])] for k in old['train_keys']]
        sets=[{r['code'] for r in rr} for rr in [train,old['test'],old['unused_calibration_keys']]]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        bycode={r['code']:r for r in train};inner={}
        for split in record['inner_splits']:
            fold=split['fold'];fitting=[bycode[c] for c in split['training_codes']];valid=[bycode[c] for c in split['validation_codes']]
            assert {r['code'] for r in valid}=={r['code'] for r in train if partition(r['code'],record['partition'])==fold}
            assert not ({r['code'] for r in fitting}&{r['code'] for r in valid})
            names=sorted({k for r in fitting for k in features(r,family)})
            assert names==split['feature_names']
            raw=np.array([[features(r,family).get(k,0.) for k in names] for r in fitting])
            scale=raw.std(axis=0);scale[scale==0]=1
            np.testing.assert_allclose(raw.mean(axis=0),split['mean'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(scale,split['scale'],rtol=1e-12,atol=1e-12)
            inner[fold]=(pairs(fitting,matrix(fitting,family,split),family),pairs(valid,matrix(valid,family,split),family))
        fit=record['fit'];candidates=fit['inner_candidates']
        assert len(candidates)==11
        for candidate in candidates:
            penalty=candidate['penalty']
            if penalty not in cfg['new_regularization']:
                original=next(c for c in old['fits'][family]['inner_candidates'] if c['penalty']==penalty)
                assert candidate==original
                continue
            total=0.;resolved=0
            for fold in candidate['folds']:
                (d,y),(vd,vy)=inner[fold['fold']];w=np.array(fold['coefficients'])
                g,n=check_fit(d,y,w,penalty,fold['optimizer']);max_grad=max(max_grad,g);max_norm=max(max_norm,n)
                answer=np.sign(np.sum(vd*w,axis=1))
                wrong=float(np.sum((answer!=vy)&(answer!=0))+.5*np.sum(answer==0))
                assert wrong==fold['wrong_plus_half_ties'] and len(vy)==fold['resolved'] and len(y)==fold['training_pairs']
                total+=wrong;resolved+=len(vy);inner_checks+=1
            assert candidate['wrong_plus_half_ties']==total and candidate['resolved']==resolved and candidate['loss']==total/resolved
        selected=min(candidates,key=lambda c:(c['loss'],-c['penalty']))['penalty']
        assert selected==fit['selected_penalty']
        d,y=pairs(train,matrix(train,family,fit),family)
        g,n=check_fit(d,y,fit['coefficients'],selected,fit['optimizer']);max_grad=max(max_grad,g);max_norm=max(max_norm,n);final_checks+=1
        loss_gains.append(record['original_best_inner_loss']-record['expanded_best_inner_loss'])
    result=dict(selection_manifest_sha256=digest(path),verifier_sha256=digest(__file__),
                all_new_inner_counts_and_cached_losses_match=True,all_authority_and_feature_transform_checks_pass=True,
                new_inner_fit_checks=inner_checks,selected_final_fit_checks=final_checks,
                maximum_gradient_inf=max_grad,maximum_coefficient_norm=max_norm,
                models_with_strict_inner_loss_improvement=sum(x>0 for x in loss_gains),
                mean_inner_loss_improvement=float(np.mean(loss_gains)),max_inner_loss_improvement=max(loss_gains),
                no_new_outer_scoring=True,scope='Selection/numerical validation only; no outer result or external scientific confirmation')
    (OUT/'selection-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
