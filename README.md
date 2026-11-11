# Aggregate comparison evaluation

Research software for testing which people-count comparisons survive conversion
from published event counts. It evaluates count fit, comparison errors, answer
yield and assurance scope together. Adult requests and child episodes/referrals
come from two separately collected English administrative systems sharing local
authorities. These are workload counts, not care need, quality or individual records.

The information restriction tests a hypothetical reuse choice: when a service has
event totals but people counts only for other areas or earlier years, should it
decline a comparison, apply a simple conversion or fit a contextual model?
Withholding published people counts makes that counterfactual evaluable; it does
not document deployed missing-data reuse. The protocol compares those choices on
equally weighted retained within-fold area pairs and specified annual changes,
not an observed distribution of analysts' requests or operational utility.

## Setup and tests

Tested on CPython 3.9.6, macOS arm64, with the 17 runtime versions in
`requirements-comparison.lock`. Other platforms are unverified. From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip --isolated --disable-pip-version-check install --only-binary=:all: -r requirements-comparison.lock
.venv/bin/python -m pip --isolated check
make test PYTHON=.venv/bin/python
make test-comparison PYTHON=.venv/bin/python
```

Core exact-arithmetic utilities use only the standard library. Comparison tests
add leakage, grouped feature selection, finite ranks, interval ties, calibration
scope, order accounting, numerical optimisation and input-boundary checks.
`src/countviews/series.py` also provides finite-history aggregate rewrites and
checked population obstructions. State indices and witness multiplicities must
be integers; Boolean and fractional values are rejected.

## Selected figures, tables and results

The actual selected outputs are included under [selected/](selected/README.md):
five figures in both PDF and PNG, six readable CSV tables, named numerical metrics
and six scientific summary groups. See [figures](selected/figures/),
[tables](selected/tables/) and [results](selected/results/). The numerical reference
payload in `experiments/comparison/config/expected.json` is also retained.
These findings/displays are distinct from provider input datasets.

To regenerate the displays without inputs or refitting, use a new directory:

```sh
PYTHONPATH=src:experiments/comparison .venv/bin/python experiments/comparison/render_selected.py --output-dir /path/to/new-render
```

The selected snapshot stays intact. Full fitting below writes separate root working
outputs, so a fresh checkout can still run the complete analysis.

## One result from numbers to figure

The 2023 child example separates changing predictions from changing the cutoff.
From the repository root, inspect the included numerical record:

```sh
.venv/bin/python - <<'PYCODE'
import json
from pathlib import Path

path = Path("selected/results/comparison-mechanism/summary.json")
group = json.loads(path.read_text())["groups"]["DfE:current:2023"]
cells = {(row["margin"], row["threshold"]): row
         for row in group["crossed_guards"]}
path_keys = [("lagged_factor", "lagged_factor"),
             ("context_forest", "lagged_factor"),
             ("context_forest", "context_forest")]
print({"resolved": group["resolved"],
       "answers": [cells[key]["answered"] for key in path_keys],
       "errors": [cells[key]["wrong"] for key in path_keys],
       "cutoffs": group["q_by_case"]["DfE-current-2023"]})
PYCODE
```

This prints 302 resolved queries, answers `[154, 149, 181]` and errors
`[0, 0, 0]`. Each key pairs the model supplying the predicted margin with the
model supplying the cutoff. The cutoffs are about 0.06256 for the lagged factor
and 0.04561 for the forest. METHODS.md defines the score and quantile rule.
The fourth cell, lagged margins with the forest cutoff, gives 190 answers.
Mixed cells are diagnostic controls without a coverage guarantee.

The matching display is `selected/figures/reuse-example.pdf` (and `.png`).
Run the display-only command above to regenerate it under the new directory's
`figures/`, with the numbers in `tables/mechanisms.csv`. To recompute the
analysis from provider inputs, run the full reproduction below, then change the
recipe's path to `results/comparison-mechanism/summary.json`. Compare the new
values with the selected record; the reference snapshot is not independent
confirmation. The full run writes the corresponding `figures/reuse-example.pdf`
and `tables/mechanisms.csv` at the repository root.

## Provider inputs

Read `data/README.md` for the five original DHSC/DfE download links, studied
versions, selections, access conditions and preprocessing. Download directly:

```sh
.venv/bin/python experiments/comparison/acquire.py --download
.venv/bin/python experiments/comparison/acquire.py --verify
```

The five CSVs total 17,071,519 bytes. No raw or prepared provider input data is
bundled in Git or project release assets. Manual provider downloads and existing
provider copies are supported as described in the data README. A revised or
unavailable provider version is reported rather than silently substituted.

## Reproduction

In a fresh checkout, after installing dependencies and verifying inputs:

```sh
make reproduce-comparison PYTHON=.venv/bin/python
```

The runner checks inputs and tests, fits the original count panel and 35-case
partition/temporal/vintage panel, reconstructs ridge fits with scalar arithmetic,
fits the original and expanded ranking grids, freezes inner selections before
outer scoring, regenerates mechanism and calibration-scope diagnostics, checks
all selected numerical payloads and produces five figures and six CSV tables.
It forces one numerical thread and retains warnings in `records/*.log`.
Existing comparison result directories, plots, tables or nonempty records are
refused before any stage starts: use a fresh checkout for another run.
The ranking grids involve 900 original and 1,080 additional inner fits; full
reproduction took about 45 seconds on the tested host and uses a few hundred MB beyond
the environment. Actual runtime and platform are recorded in `records/reproduction.json`.

`results/` holds detailed regenerated cases and summaries, `figures/` the five
plots, and `tables/` plain CSV comparisons and named metrics. Counts and labels
must match exactly; floating results use relative tolerance 1e-8 and absolute
1e-10. The compact expected payloads are fixed-data references, not held-out
confirmation. Dense predictions, raw inputs, environments and caches stay out of
Git. The selected scientific displays and summaries remain tracked under
`selected/`; dense working outputs are regenerated locally. No external service is used after acquisition.

Better count fit need not improve comparisons. Models can correct some orders and introduce new errors;
temporal preferences change by year. Specified-family calibration gives up joint
protection. Exchangeability and future transport are unestablished; zero observed
errors do not bound future risk. See `METHODS.md` for assumptions and limitations.

Original code/docs use MIT; original result payloads and generated displays use
CC BY 4.0. Covered government information retains OGL v3.0 terms and attribution.
See `LICENSING.md`, `LICENSES/` and `THIRD-PARTY.txt`. No dependency binaries are bundled.
