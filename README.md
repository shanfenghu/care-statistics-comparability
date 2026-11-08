# Aggregate comparison evaluation

Research software for testing which people-count comparisons survive conversion
from published event counts. It evaluates count fit, comparison errors, answer
yield and assurance scope together. Adult requests and child episodes/referrals
come from two separately collected English administrative systems sharing local
authorities. These are workload counts, not care need, quality or individual records.

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

## Exact inputs

Read `data/README.md` before acquisition. Five exact CSVs total 17,071,519 bytes;
their SHA-256 values and source URLs are in `data/comparison-input-manifest.json`.
They are supplied separately from Git. Given the retained versioned input archive:

```sh
.venv/bin/python experiments/comparison/acquire.py --archive /path/to/aggregate-input-snapshot-1.0.2.tar.gz
```

Alternatively use `--from-dir /path/to/retained-inputs` with the manifest-relative
paths, or explicitly request `--download`. A changed publisher download fails
verification; it never becomes a replacement benchmark version.

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
Existing comparison outputs are refused: use a fresh checkout for another run.
The ranking grids involve 900 original and 1,080 additional inner fits; full
reproduction took 43 seconds on the tested host and uses a few hundred MB beyond
the environment. Actual runtime and platform are recorded in `records/reproduction.json`.

`results/` holds detailed regenerated cases and summaries, `figures/` the five
plots, and `tables/` plain CSV comparisons and named metrics. Counts and labels
must match exactly; floating results use relative tolerance 1e-8 and absolute
1e-10. The compact expected payloads are fixed-data references, not held-out
confirmation. Dense predictions, raw inputs, environments and caches stay out of
Git; all are regenerated locally. No external service is used after acquisition.

Better count fit need not improve comparisons. Repairs can create new errors;
temporal preferences change by year. Specified-family calibration gives up joint
protection. Exchangeability and future transport are unestablished; zero observed
errors do not bound future risk. See `METHODS.md` for assumptions and limitations.

Original code/docs use MIT; original result payloads and generated displays use
CC BY 4.0. Covered government information retains OGL v3.0 terms and attribution.
See `LICENSING.md`, `LICENSES/` and `THIRD-PARTY.txt`. No dependency binaries are bundled.
