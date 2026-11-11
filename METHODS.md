# Comparison methods and interpretation

A target is one adult age band or one child measure (episodes or referrals).
Configuration fields call a target a `family`; it does not mean the adult or
child source. A native count is the published people count. Its native bin
represents publication rounding, while a predictive interval represents an
allowance around a model estimate. A reference is resolved when its bins
establish a strict direction.

Native published people counts are withheld test references, not missing values
or administrative truth. Geographic queries compare two authorities within the
same target family, year, release and test fold. Temporal queries compare the
current count with the known prior native anchor. Overlapping native publication
bins remain unresolved. Point ties abstain; wrong/answered error and resolved/all
query coverage have different denominators. Retained within-fold geographic pairs
have equal weight; the workload does not estimate analysts' demand or operational
utility. Withholding published counterparts tests a hypothetical restricted-information
reuse choice, rather than documenting missingness in these releases.

Literal, pooled family factor, monotone isotonic, ridge and random-forest count
converters share specified information access. Each chooses how much of it to
use; this does not equate their effective features. Geographic test native labels,
their aggregate totals and their historical people counts are excluded. Temporal
models may use the prior people/event ratio, event growth and year. The lagged
factor is a strong historical reference; its change sign equals event growth.
Current people/event ratios enter post-prediction diagnostics, never selection.

Ridge and forests model log people/event ratios. Features use event totals,
cross-family event ratios and region; transformations fit only training data.
Forests use 100 trees and minimum leaf size five. Isotonic endpoint clipping and
its small-count failures are retained. Pairwise logistic scores use grouped
inner validation and provide order, not counts or calibrated intervals.
Configuration files fix hyperparameters, seeds, folds, grids and budgets.
Unknown count-estimator names, non-temporal lagged-factor calls and invalid
calibration levels or nonconformity scores are rejected rather than substituted.
Answer budgets must be fractions in [0,1]; invalid fractions are rejected.

Five-fold geographic rotations train on three authority folds, calibrate on one
and test on one; all families move together. Two additional deterministic salts
change both fits and within-fold query sets. Temporal targets are 2020/2023/2025:
train through target minus two, calibrate on the prior year, test on target.
The selected 2024/2025 release fields match; this measured null does not imply
general revision stability. Shared authorities and repeated years are dependent.

Native-bin coverage clips the published lower endpoint at zero; the calibration
score remains conservative when a bin is truncated. Calibration divides
native-bin discrepancies by event totals. A corrected finite rank can be
infinite; an unavailable rank is never replaced by the maximum.
Each model learns its own error allowance from labelled calibration areas at the
same nominal level. For each calibration authority, compute each target's
absolute prediction error plus its people-count rounding radius, then divide
by that target's event total. The joint score is the larger of the two values.
With n calibration authorities and nominal miscoverage alpha, use rank
`ceil((n + 1) * (1 - alpha))`. The cutoff is the score at that rank in ascending order, or
infinity when the rank exceeds n. For a prediction p and event count E, the
range is from `max(0, p - cutoff * E)` to `p + cutoff * E`. Area ranges must
strictly separate; an annual-change range must exclude the known prior people
count. The default score protects both adult age-band targets or
both child episode/referral targets together. Separated predicted intervals support
an answer. Joint-vector and specified-family
scores protect different events; Bonferroni has sample-size costs. Under the stated
exchangeable law, a specified pair's selective-risk bound divides a failure bound
by its answering-and-reference-resolution probability. Observed workload coverage
cannot replace that probability. Spatial exchangeability and temporal transport
are not established. Fixed-budget selectors and crossed controls have no conformal
guarantee.

All outcomes remain recorded: errors, abstentions, infinite intervals, isotonic
failures, ranking boundary choices and warnings. Corrections and new errors compare common
point answers; ties and own-answer denominators are accounted separately.
Crossed margins/thresholds are mechanical diagnostics, not causal attribution or
evidence that calibration gains are unrelated to predictor quality. The 2023 child
illustration holds the lagged cutoff fixed while switching predictions (154 to
149 answers), then switches only to the forest's calibrated cutoff (181). The
alternative path is 154 to 190 to 181; both own policies use nominal 95% joint
coverage, while crossed policies inherit no coverage guarantee.
Two hundred seeded authority-weight replicates condition on fits/cutoffs/folds;
they do not certify spatial 95-percent confidence or all learning uncertainty.
A zero-error percentile interval cannot bound future selected-answer risk.

## Counting-rule control

The counting rule gives a separate check when each person contributes at least
one event in the same population and period. The event total plus its
publication radius is an upper bound on people. A decline is certified if
that upper bound is strictly below the exact prior people count. This check
uses the observed counts and anchor, without a model or calibrated cutoff.

After the full reproduction, run this from the repository root to recover the
2025 child temporal control. The case's literal rows carry the original event
counts and native references. The native current count is used only to score
the certificate.

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path

case = Path("results/comparison-panel/DfE-forward-2025.json")
rows = json.loads(case.read_text())["predictions"]["literal"]
resolved = [row for row in rows
            if abs(row["p"] - row["previous_p"]) > row["radius"]]
declines = [row for row in resolved
            if row["e"] + row["radius"] < row["previous_p"]]
wrong = sum(row["p"] + row["radius"] >= row["previous_p"] for row in declines)
print({"resolved": len(resolved), "certified_declines": len(declines), "wrong": wrong})
PY
```

The recorded counts are 302 resolved, 66 certified declines and zero errors.

## Exact finite-state utilities

The finite-state utilities use exact rational linear algebra and classical
weighted-automaton forward-space saturation. They expose constructive population
obstructions; they do not infer programmes from prose or reconstruct care records.
Methodological foundations remain credited in module docstrings. Reproduction
checks implementation agreement on fixed development inputs, not independent
scientific validation, future accuracy or deployment suitability.
