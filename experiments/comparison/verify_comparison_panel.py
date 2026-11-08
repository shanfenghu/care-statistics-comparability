"""Numerical and policy checks for verify comparison panel."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import json
import math
import warnings

import numpy as np

from comparison_panel import CONFIG, OUT, ROOT, digest, features, load_rows


def solve(a, b):
    """Small dense system by scalar partial-pivot elimination."""
    a = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = len(b)
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(a[i][j]))
        a[j], a[pivot] = a[pivot], a[j]
        assert abs(a[j][j]) > 1e-12
        for i in range(j+1, n):
            factor = a[i][j]/a[j][j]
            for k in range(j, n+1):a[i][k] -= factor*a[j][k]
    x = [0.0]*n
    for j in range(n-1, -1, -1):
        x[j] = (a[j][n]-math.fsum(a[j][k]*x[k] for k in range(j+1,n)))/a[j][j]
    return x


def scalar_ridge(train, evaluate, family, alpha, temporal):
    """Rebuild scaling/fit/prediction from scalar sums, without sklearn or BLAS."""
    ff = [features(r, family, temporal) for r in train]
    keys = sorted({k for f in ff for k in f}); n = len(train)
    x = [[f.get(k,0.0) for k in keys] for f in ff]
    means = [math.fsum(row[j] for row in x)/n for j in range(len(keys))]
    scale = [math.sqrt(math.fsum((row[j]-means[j])**2 for row in x)/n) for j in range(len(keys))]
    scale = [s if s > 1e-14 else 1.0 for s in scale]
    z = [[(v-means[j])/scale[j] for j,v in enumerate(row)] for row in x]
    y = [math.log(r["p"][family]/r["e"][family]) for r in train]; mean_y = math.fsum(y)/n
    a = [[math.fsum(row[j]*row[k] for row in z)+(alpha if j==k else 0) for k in range(len(keys))] for j in range(len(keys))]
    b = [math.fsum(row[j]*(yy-mean_y) for row,yy in zip(z,y)) for j in range(len(keys))]
    coef = solve(a,b)
    answer = []
    for r in evaluate:
        f=features(r,family,temporal)
        eta=mean_y+math.fsum((f.get(k,0.0)-means[j])/scale[j]*coef[j] for j,k in enumerate(keys))
        answer.append(r["e"][family]*math.exp(eta))
    return answer


def interval_order(lo, hi, other_lo, other_hi):
    return 1 if lo > other_hi else -1 if hi < other_lo else 0


def main():
    config=json.loads(CONFIG.read_text()); rows,_,_=load_rows(config)
    summary=json.loads((OUT/"summary.json").read_text())
    assert summary["config_sha256"] == digest(CONFIG)
    assert summary["script_sha256"] == digest(ROOT/"experiments/comparison/comparison_panel.py")
    max_abs=0.0; max_relative=0.0; fits=0; query_totals=defaultdict(lambda:dict(queries=0,resolved=0,answered=0,wrong=0))
    for info in summary["cases"]:
        p=ROOT/info["path"]; assert digest(p)==info["sha256"]
        case=json.loads(p.read_text()); source=case["source"]; temporal=case["task"]=="forward"
        sets={k:[rows[(source,r["year"],r["code"])] for r in v] for k,v in case["partitions"].items()}
        for f in range(2):
            scalar=scalar_ridge(sets["train"],sets["cal"]+sets["test"],f,config["ridge_alpha"],temporal)
            saved=[r for r in case["predictions"]["context_ridge"] if r["family"]==f]
            bykey={(r["code"],r["year"]):r["prediction"] for r in saved}
            for r, v in zip(sets["test"],scalar[len(sets["cal"]):]):
                other=bykey[(r["code"],r["year"])];err=abs(v-other)
                max_abs=max(max_abs,err);max_relative=max(max_relative,err/max(1,abs(other)))
                assert err <= 1e-8*max(1,abs(other))
            fits+=1
        # Reconstruct every point/guarded comparison using plain scalar interval
        # logic, independently of vectorized query construction and aggregation.
        for method, predictions in case["predictions"].items():
            for f in range(2):
                rr=[r for r in predictions if r["family"]==f]
                pairs=((r,None) for r in rr) if temporal else combinations(rr,2)
                for a,b in pairs:
                    truth=interval_order(a["p"]-a["radius"],a["p"]+a["radius"],
                                         b["p"]-b["radius"] if b else a["previous_p"],
                                         b["p"]+b["radius"] if b else a["previous_p"])
                    for mode in ("point","guard"):
                        if mode=="point":
                            direction=interval_order(a["prediction"],a["prediction"],b["prediction"] if b else a["previous_p"],b["prediction"] if b else a["previous_p"])
                        else:
                            direction=interval_order(a["lower"],a["upper"] if a["upper"] is not None else math.inf,b["lower"] if b else a["previous_p"],(b["upper"] if b["upper"] is not None else math.inf) if b else a["previous_p"])
                        total=query_totals[(source+":"+case["task"],method+":"+mode)]
                        total["queries"]+=1;total["resolved"]+=bool(truth)
                        total["answered"]+=bool(truth and direction);total["wrong"]+=bool(truth and direction and truth!=direction)
    for (group,method),total in query_totals.items():
        s=summary["summaries"][group]["methods"][method]
        assert [total[k] for k in ("queries","resolved","answered","wrong")] == [s[k] for k in ("queries","resolved_queries","answers_on_resolved","wrong_on_resolved")]
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        eye=np.identity(15); product=eye@eye
    assert np.array_equal(product,eye)
    result=dict(ridge_fits_checked=fits, max_absolute_prediction_difference=max_abs,
                max_relative_prediction_difference=max_relative,
                independent_scalar_query_aggregates=len(query_totals),
                identity_matmul_exact=True, identity_matmul_warnings=sorted({str(w.message) for w in ws}),
                warning_context="Main run emitted matmul RuntimeWarnings; independent scalar ridge and all-query replay passed. See NumPy issue 28687; no numerical warnings were silently discarded from the main run.",
                primary_issue="https://github.com/numpy/numpy/issues/28687",
                source_summary_sha256=digest(OUT/"summary.json"), verifier_sha256=digest(__file__))
    (OUT/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
