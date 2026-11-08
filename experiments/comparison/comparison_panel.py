"""Aggregate converter fitting and comparison evaluation; native test labels never enter features."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import time

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from countviews.values import parse_count
from input_readers import fold, read_pairs

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/comparison/config/panel.json"
OUT = ROOT / "results/comparison-panel"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(config):
    """Read only the frozen families and preserve a complete-family inventory."""
    old = json.loads((ROOT / "experiments/comparison/config/sources.json").read_text())
    sources = []
    for item in old["sources"]:
        item = dict(item)
        if item["format"] == "adult_requests":
            sources.append(item)
        else:
            item["pairs"] = [p for p in item["pairs"] if p[0] in config["families"]["DfE"]]
            if item["pairs"]:
                sources.append(item)
    pairs, provenance = read_pairs(dict(sources=sources))
    regions = {}
    for item in sources:
        source = "DHSC" if item["format"] == "adult_requests" else "DfE"
        with (ROOT / item["path"]).open(encoding="utf-8-sig") as stream:
            for r in csv.DictReader(stream):
                adult = source == "DHSC"
                if r.get("GeographyLevel" if adult else "geographic_level") != ("Local Authority" if adult else "Local authority"):
                    continue
                key = (source, int(r["FYEnding" if adult else "time_period"]), r["GeographyCode" if adult else "new_la_code"])
                region = r["RegionGOCode" if adult else "region_code"]
                assert key not in regions or regions[key] == region
                regions[key] = region
    buckets = defaultdict(dict)
    for r in pairs:
        if r["level"] != "Local authority" or r["family"] not in config["families"][r["source"]]:
            continue
        key = (r["source"], int(r["period"]), r["code"])
        assert r["family"] not in buckets[key]
        buckets[key][r["family"]] = r
    rows, inventory = {}, []
    for key, values in sorted(buckets.items()):
        source, year, code = key
        families = config["families"][source]
        counts = [(parse_count(values[f]["events"]).value, parse_count(values[f]["persons"]).value)
                  if f in values else (None, None) for f in families]
        reason = "eligible"
        if any(x is None or x <= 0 for pair in counts for x in pair):
            reason = "missing_suppressed_unreliable_or_nonpositive"
        inventory.append(dict(source=source, year=year, code=code, status=reason))
        if reason != "eligible":
            continue
        radius = config["reference_bins"][source]
        assert regions[key]
        row = dict(source=source, year=year, code=code, region=regions[key],
                   e=[x[0] for x in counts], p=[x[1] for x in counts], radius=radius)
        row["event_below_person"] = any(e + radius < p - radius for e, p in counts)
        rows[key] = row
    for (source, year, code), row in rows.items():
        row["previous"] = rows.get((source, year - 1, code))
    return rows, inventory, provenance


def features(row, family, temporal=False):
    """Only observable event context and, for temporal reuse, prior native labels."""
    e = row["e"][family]
    result = {"log_events": math.log(e), "other_event_ratio": math.log(row["e"][1-family]/e),
              "region=" + row["region"]: 1.0}
    if temporal:
        previous = row["previous"]
        assert previous["year"] == row["year"] - 1
        result.update(year=row["year"]-2014, lagged_ratio=math.log(previous["p"][family]/previous["e"][family]),
                      event_growth=math.log(e/previous["e"][family]))
    return result


def fit_predict(method, train, evaluate, family, config, temporal):
    """Return a fixed estimator's predictions without using evaluate targets."""
    e = np.array([r["e"][family] for r in train], dtype=float)
    p = np.array([r["p"][family] for r in train], dtype=float)
    ee = np.array([r["e"][family] for r in evaluate], dtype=float)
    if method == "literal":
        return ee
    if method == "family_factor":
        return ee * p.sum()/e.sum()
    if method == "lagged_factor":
        assert temporal
        return np.array([r["e"][family]*r["previous"]["p"][family]/r["previous"]["e"][family] for r in evaluate])
    if method == "isotonic":
        return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(e, p).predict(ee)
    vector = DictVectorizer(sparse=False)
    x = vector.fit_transform([features(r, family, temporal) for r in train])
    xx = vector.transform([features(r, family, temporal) for r in evaluate])
    model = (make_pipeline(StandardScaler(), Ridge(alpha=config["ridge_alpha"])) if method == "context_ridge"
             else RandomForestRegressor(**config["forest"], random_state=config["seed"]))
    model.fit(x, np.log(p/e))
    return ee * np.exp(model.predict(xx))


def conformal_quantile(scores, alpha):
    """Finite-sample rank correction, including legitimately infinite intervals."""
    rank = math.ceil((len(scores)+1)*(1-alpha))
    return float(np.sort(scores)[rank-1]) if rank <= len(scores) else math.inf


def conformal_bounds(predictions, rows, alpha):
    """Calibrate entire authority vectors, including every native-bin endpoint."""
    p = np.array([r["p"] for r in rows], dtype=float)
    e = np.array([r["e"] for r in rows], dtype=float)
    radius = np.array([r["radius"] for r in rows])[:, None]
    scores = np.max((np.abs(p-predictions)+radius)/e, axis=1)
    return conformal_quantile(scores, alpha)


def order(lo_a, hi_a, lo_b, hi_b):
    """Strict interval order; touching or overlapping intervals abstain."""
    return np.where(lo_a > hi_b, 1, np.where(hi_a < lo_b, -1, 0))


def case_results(case, method, test, prediction, q, temporal):
    """Store predictions sufficient to reconstruct every fixed comparison query."""
    e = np.array([r["e"] for r in test], dtype=float)
    p = np.array([r["p"] for r in test], dtype=float)
    radius = np.array([r["radius"] for r in test])[:, None]
    lower, upper = np.maximum(0, prediction-q*e), prediction+q*e
    covered = (lower <= p-radius) & (upper >= p+radius)
    rows = []
    for i, r in enumerate(test):
        for f in range(2):
            rows.append(dict(case=case, method=method, code=r["code"], year=r["year"], family=f,
                             e=int(e[i, f]), p=int(p[i, f]), radius=r["radius"], prediction=float(prediction[i, f]),
                             lower=float(lower[i, f]), upper=float(upper[i, f]) if math.isfinite(upper[i, f]) else None,
                             covered=bool(covered[i, f]), group_covered=bool(covered[i].all()),
                             previous_p=r["previous"]["p"][f] if temporal else None))
    query_arrays = []
    for f in range(2):
        if temporal:
            ia = np.arange(len(test)); ib = np.full(len(test), -1)
            anchor = np.array([r["previous"]["p"][f] for r in test])
            truth = order(p[:, f]-radius[:, 0], p[:, f]+radius[:, 0], anchor, anchor)
            point = order(prediction[:, f], prediction[:, f], anchor, anchor)
            guarded = order(lower[:, f], upper[:, f], anchor, anchor)
        else:
            ia, ib = np.array(list(combinations(range(len(test)), 2)), dtype=int).T
            truth = order(p[ia, f]-radius[ia, 0], p[ia, f]+radius[ia, 0], p[ib, f]-radius[ib, 0], p[ib, f]+radius[ib, 0])
            point = order(prediction[ia, f], prediction[ia, f], prediction[ib, f], prediction[ib, f])
            guarded = order(lower[ia, f], upper[ia, f], lower[ib, f], upper[ib, f])
        query_arrays.append(dict(family=f, ia=ia, ib=ib, truth=truth, point=point, guard=guarded))
    return rows, query_arrays


def ratio(a, b):
    return float(a/b) if b else None


def evaluate_group(cases, config):
    """Aggregate dependent queries and resample whole authorities, not pairs."""
    codes = sorted({r["code"] for c in cases for r in c["test"]})
    index = {c:i for i,c in enumerate(codes)}
    models = sorted({m for c in cases for m in c["results"]})
    stash, summaries = {}, {}
    for method in models:
        rr, qq = [], []
        for c in cases:
            rows, queries = c["results"][method]
            rr.extend(rows)
            for q in queries:
                qq.append((q, np.array([index[c["test"][i]["code"]] for i in q["ia"]]),
                           np.array([index[c["test"][i]["code"]] if i >= 0 else -1 for i in q["ib"]])))
        ri = np.array([index[r["code"]] for r in rr])
        native = np.array([r["p"] for r in rr], dtype=float)
        ae = np.abs(np.array([r["prediction"] for r in rr])-native)
        covered = np.array([r["covered"] for r in rr]); gc = np.array([r["group_covered"] for r in rr])
        qi = np.concatenate([v[1] for v in qq]); qj = np.concatenate([v[2] for v in qq])
        truth = np.concatenate([v[0]["truth"] for v in qq])
        for mode in ("point", "guard"):
            pred = np.concatenate([v[0][mode] for v in qq]); resolved = truth != 0; answered = pred != 0
            wrong = answered & resolved & (pred != truth)
            key = method + ":" + mode
            stash[key] = (ri, ae, native, covered, gc, qi, qj, resolved, answered, wrong)
    def compute(data, weights):
        ri, ae, native, covered, gc, qi, qj, resolved, answered, wrong = data
        rw = weights[ri]; qw = weights[qi]*np.where(qj >= 0, weights[np.maximum(qj, 0)], 1)
        return dict(mape=100*ratio(np.sum(rw*ae/native), rw.sum()), wape=100*ratio(np.sum(rw*ae), np.sum(rw*native)),
                    row_bin_coverage=ratio(np.sum(rw*covered), rw.sum()), authority_vector_coverage=ratio(np.sum(rw*gc), rw.sum()),
                    queries=len(qi), resolved_queries=int(resolved.sum()), answers_on_resolved=int((answered & resolved).sum()),
                    wrong_on_resolved=int(wrong.sum()), coverage_all=ratio(np.sum(qw*answered), qw.sum()),
                    coverage_resolved=ratio(np.sum(qw*answered*resolved), np.sum(qw*resolved)),
                    risk=ratio(np.sum(qw*wrong), np.sum(qw*answered*resolved)))
    unit = np.ones(len(codes)); rng = np.random.default_rng(config["seed"])
    bootstrap = {key:[] for key in stash}
    for _ in range(config["bootstrap_authority_repetitions"]):
        w = rng.multinomial(len(codes), np.repeat(1/len(codes), len(codes)))
        for key, data in stash.items():
            bootstrap[key].append(compute(data, w))
    for key, data in stash.items():
        value = compute(data, unit)
        value["conditional_authority_bootstrap_95"] = {}
        for metric in ("mape", "wape", "risk", "coverage_resolved", "row_bin_coverage"):
            vals = [v[metric] for v in bootstrap[key] if v[metric] is not None]
            value["conditional_authority_bootstrap_95"][metric] = np.quantile(vals, [.025, .975]).tolist() if vals else None
        summaries[key] = value
    contrasts = {}
    for key in stash:
        base = "family_factor:"+key.split(":")[1]
        if key == base:
            continue
        contrasts[key+" minus "+base] = {}
        for metric in ("mape", "risk", "coverage_resolved"):
            ds = [a[metric]-b[metric] for a,b in zip(bootstrap[key], bootstrap[base]) if a[metric] is not None and b[metric] is not None]
            contrasts[key+" minus "+base][metric] = np.quantile(ds, [.025, .975]).tolist() if ds else None
    return dict(authorities=len(codes), methods=summaries, paired_contrasts_95=contrasts,
                reference_control=dict(coverage_resolved=1, risk=0, current_native_values_required=True),
                reject_all_control=dict(coverage_resolved=0, risk=None),
                bootstrap_scope="Conditional on fitted models and fixed folds; authority weights shared by both families and query endpoints")


def main():
    config = json.loads(CONFIG.read_text()); OUT.mkdir(exist_ok=True)
    rows, inventory, provenance = load_rows(config)
    cases = []
    for source in config["families"]:
        available = [r for (s, year, _), r in rows.items() if s == source and year == config["geographic_year"]]
        for j in range(config["folds"]):
            train = [r for r in available if fold(r["code"]) not in (j, (j+1)%5)]
            cal = [r for r in available if fold(r["code"]) == (j+1)%5]
            test = [r for r in available if fold(r["code"]) == j]
            assert not ({r["code"] for r in train} & {r["code"] for r in cal+test})
            assert not ({r["code"] for r in cal} & {r["code"] for r in test})
            cases.append(dict(name=f"{source}-geographic-{j}", source=source, task="geographic", train=train, cal=cal, test=test))
    temporal = [r for (s, _, _), r in rows.items() if s == "DfE" and r["previous"] is not None]
    cases.append(dict(name="DfE-forward-2025", source="DfE", task="forward",
                      train=[r for r in temporal if 2014 <= r["year"] <= 2023],
                      cal=[r for r in temporal if r["year"] == 2024],
                      test=[r for r in temporal if r["year"] == 2025]))
    manifest = []
    for case in cases:
        temporal = case["task"] == "forward"; methods = config["models"] + ([config["temporal_extra_model"]] if temporal else [])
        case["results"] = {}; timings = {}; qs = {}
        train, cal, test = case["train"], case["cal"], case["test"]
        assert all(len(v) >= 2 for v in (train, cal, test))
        for method in methods:
            start = time.perf_counter()
            both = np.column_stack([fit_predict(method, train, cal+test, f, config, temporal) for f in range(2)])
            assert np.isfinite(both).all()
            q = conformal_bounds(both[:len(cal)], cal, config["alpha"])
            case["results"][method] = case_results(case["name"], method, test, both[len(cal):], q, temporal)
            timings[method] = time.perf_counter()-start; qs[method] = q if math.isfinite(q) else None
        saved = dict(case=case["name"], source=case["source"], task=case["task"],
                     partitions={k:[dict(code=r["code"], year=r["year"]) for r in case[k]] for k in ("train","cal","test")},
                     q=qs, fit_predict_seconds=timings,
                     predictions={m:v[0] for m,v in case["results"].items()})
        path = OUT/(case["name"]+".json"); path.write_text(json.dumps(saved, indent=2, allow_nan=False)+"\n")
        manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path)))
    summaries = {}
    for source, task in sorted({(c["source"], c["task"]) for c in cases}):
        selected = [c for c in cases if c["source"] == source and c["task"] == task]
        summaries[source+":"+task] = evaluate_group(selected, config)
    # Family-specific summaries prevent favorable pooling across the two strata.
    family_summaries = {}
    for source, task in sorted({(c["source"], c["task"]) for c in cases}):
        for f, family in enumerate(config["families"][source]):
            selected = []
            for c in cases:
                if (c["source"],c["task"]) != (source,task):continue
                clone = dict(c); clone["results"] = {m:([r for r in rr if r["family"] == f], [q for q in qq if q["family"] == f]) for m,(rr,qq) in c["results"].items()}; selected.append(clone)
            family_summaries[source+":"+task+":"+family] = evaluate_group(selected, config)
    result = dict(config_sha256=digest(CONFIG), script_sha256=digest(__file__), raw_inputs=provenance,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                  inventory=dict(counts=[dict(source=s,year=y,status=st,count=n) for (s,y,st),n in sorted(Counter((r["source"],r["year"],r["status"]) for r in inventory).items())],
                                 event_below_person_authority_years=sum(r["event_below_person"] for r in rows.values())),
                  cases=manifest, summaries=summaries, family_summaries=family_summaries)
    (OUT/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for task, result in summaries.items():
        print(task, result["authorities"], {m:{k:round(v[k],4) if v[k] is not None else None for k in ("mape","risk","coverage_resolved","row_bin_coverage")} for m,v in result["methods"].items()}, flush=True)


if __name__ == "__main__":
    main()
