# Selected scientific outputs

These actual figures, readable tables and numerical summaries are distributed
with the software. They are the selected outputs of the recorded fixed-data
comparison run, not raw provider datasets or prepared input tables. They retain
both benefits and adverse findings. The six JSON groups also retain source/version
and method provenance; relative case-file references identify the larger local
run records, which the full reproduction regenerates rather than distributing.

## Figures

PDF and PNG copies of all five selected figures are in `figures/`:

- `reuse-example`: on fixed child temporal queries, changing predictions and the
  cutoff has distinct effects on answer yield; the shown path is -5 +32 = +27.
- `reuse-fit`: better count fit need not improve point comparisons.
- `reuse-guards`: calibrated guards and matched answer budgets trade off answer
  yield and errors; they do not establish future transport.
- `reuse-temporal`: count error, comparison errors and answer yield vary by year.
- `reuse-splits`: refitted geographic partitions reveal answer-yield sensitivity.

## Tables and key results

`tables/` contains six CSVs: `geographic`, `temporal`, `vintage`, `mechanisms`,
`calibration-scope` and `cohorts`. `metrics.json` holds the named numerical results.
The six groups under `results/` retain the count panel, robustness contrasts,
original and expanded ranking results, mechanism diagnostics and calibration-scope
results. The robustness verification receipt records the scalar/numerical and
preceding-release controls used in the vintage table. The existing
`../experiments/comparison/config/expected.json` remains the regression reference.
No refitting or scientific changes were made to include these files.

## Regeneration

After the environment setup in the root README, from the repository root:

```sh
PYTHONPATH=src:experiments/comparison .venv/bin/python experiments/comparison/render_selected.py --output-dir /path/to/new-render
```

Use a new output directory. This reads the distributed `selected/results/`
summaries and regenerates all five PDF/PNG figures, six CSVs and named metrics
without downloading provider inputs, reading dense case dumps or fitting models.
It preserves this selected snapshot. An existing output directory is refused.

To repeat the full analysis, download the studied inputs from the providers as
described in `../data/README.md`, then run `make reproduce-comparison
PYTHON=.venv/bin/python` in a fresh checkout with no generated working outputs.
The full run writes root `results/`, `figures/`, `tables/` and `records/`; the
published snapshot remains separately under `selected/`. Numerical and statistical
interpretation takes priority over identical binary metadata. The recorded run
and selected displays are development evidence, not independent validation.

Original scientific results and displays use CC BY 4.0; credit Aggregate comparison
evaluation contributors. Source information retains the applicable OGL v3.0 terms:
DHSC, Adult social care activity report, England, 2024 to 2025, STS001 corrected
June 2026; DfE, Children in need B1/C1, reporting years 2025 and 2024. Contains public
sector information licensed under OGL v3.0. Crown copyright. Preserve exclusions,
third-party rights and non-endorsement conditions; see the root LICENSING.md,
LICENSES/ and THIRD-PARTY.txt.
