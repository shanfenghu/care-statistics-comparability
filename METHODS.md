# Comparison methods and interpretation

Native published people counts are withheld test references, not missing values
or administrative truth. Geographic queries compare two authorities within the
same target family, year, release and test fold. Temporal queries compare the
current count with the known prior native anchor. Overlapping native publication
bins remain unresolved. Point ties abstain; wrong/answered error and resolved/all
query coverage have different denominators.

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

Five-fold geographic rotations train on three authority folds, calibrate on one
and test on one; all families move together. Two additional deterministic salts
change both fits and within-fold query sets. Temporal targets are 2020/2023/2025:
train through target minus two, calibrate on the prior year, test on target.
The selected 2024/2025 release fields match; this measured null does not imply
general revision stability. Shared authorities and repeated years are dependent.

Calibration divides native-bin discrepancies by event totals. A corrected finite
rank can be infinite; an unavailable rank is never replaced by the maximum.
Separated predicted intervals support an answer. Joint-vector and specified-family
scores protect different events; Bonferroni has sample-size costs. Under the stated
exchangeable law, a specified pair's selective-risk bound divides a failure bound
by its answering-and-reference-resolution probability. Observed workload coverage
cannot replace that probability. Spatial exchangeability and temporal transport
are not established. Fixed-budget selectors and crossed controls have no conformal
guarantee. Counting-rule containment can certify some child declines and supplies
a different assurance from empirical prediction.

All outcomes remain recorded: errors, abstentions, infinite intervals, isotonic
failures, ranking boundary choices and warnings. Repairs/damage compare common
point answers; ties and own-answer denominators are accounted separately.
Crossed margins/thresholds are mechanical diagnostics, not causal attribution.
Two hundred seeded authority-weight replicates condition on fits/cutoffs/folds;
they do not certify spatial 95-percent confidence or all learning uncertainty.
A zero-error percentile interval cannot bound future selected-answer risk.

The finite-state utilities use exact rational linear algebra and classical
weighted-automaton forward-space saturation. They expose constructive population
obstructions; they do not infer programmes from prose or reconstruct care records.
Methodological foundations remain credited in module docstrings. Reproduction
checks implementation agreement on fixed development inputs, not independent
scientific validation, future accuracy or deployment suitability.
