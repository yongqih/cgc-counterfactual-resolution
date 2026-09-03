# CGC-RESOLUTION-2 frozen protocol

## Objective and correction of RESOLUTION-1

CGC-RESOLUTION-2 asks whether the exact same frozen M2 context-specific
counterfactual predictions retain more reproducible information after
projection to externally defined biological coordinates than in the full
25,695-gene vector, beyond dimensionality reduction and denoising controls.

RESOLUTION-1 ended as `COUNTERFACTUAL_RESOLUTION_AUDIT_INVALID` because missing
gene-vector prediction artifacts could not be regenerated under its no-replay
rule, and because direct drug-to-pathway targeting was incorrectly made a
pathway-response eligibility condition. No pathway or program outcome was
inspected in RESOLUTION-1. The prior invalid verdict is preserved.

This revision allows deterministic replay solely to materialize the missing
vectors and removes drug-target coverage from pathway eligibility. It remains
a proof of concept, not a model-capacity experiment or pathway benchmark.

## Frozen source and budgets

- Parent commit: `e47b76bd267545c45c19d4a08dd6dfc787d306e7`.
- Frozen EC results commit: `dad1ca9dc86a6e0104976026810a246c20feba52`.
- Estimator: `M2_AFFINE_RIDGE` only.
- Primary budget: `m=49,k=92`.
- Secondary budget: `m=40,k=4`.
- Target: context-specific excess response, not shared/raw drug response.
- Plates: plate 6 and plate 14 remain paired.
- Gene axis: the frozen ordered G_PRIMARY axis of 25,695 genes.

## Deterministic replay authorization and limits

Replay must load, not reselect, every target/plate/budget Ridge and rank value
from the frozen `parameter_rows_json` records. It must use the frozen split
manifest, support orders, sentinel orders, null-independent M2 code,
preprocessing, response tensor, Grams, seeds, and axes. The only allowed model
operation is the original `fit_batches` call with those frozen inputs, followed
by materialization of the resulting gene vector.

No call to `select_parameters` is permitted. No tuning, alternative estimator,
new seed, support modification, new budget, coefficient rescue, or
outcome-dependent change is permitted. Replay code must hash all sources and
write predictions before any pathway/program outcome is read.

For support contexts `S_m` and frozen M2 weights `w_p`, materialize the excess
prediction as

`Dhat[p,c] = sum_s (w[p,s] - 1/m) * Delta[p,s]`.

The paired truth is

`D[p,c] = Delta[p,c] - mean_s Delta[p,s]`.

For `(40,4)`, all eight frozen support sequences are independently replayed and
their paired utility contributions are averaged within target entry before
biological bootstrap inference, matching the original EC sufficient-statistic
semantics.

## Replay integrity hard gate

Before any lower-resolution outcome is inspected:

1. Recompute the original EC metric with the original ratio-of-sums scorer and
   reproduce `0.11068053088808294` at `(49,92)` and
   `0.07718711664021238` at `(40,4)` to absolute tolerance `1e-10`.
2. Replay a deterministic audit subset twice and require bitwise-identical
   float32 materialized vectors and identical frozen M2 weights.
3. Replace each hidden target truth with two adversarial alternatives in a
   fit-side sealed view and require bitwise-identical support identities,
   hyperparameters, weights, and predictions.
4. Require the exact frozen context/intervention/plate/gene/support/sentinel
   identities and source hashes in `RESOLUTION2_REPLAY_MANIFEST.json`.
5. Require finite paired denominators for every primary statistic.

Metric mismatch returns `RESOLUTION2_FROZEN_REPLAY_MISMATCH`. Any provenance,
axis, leakage, reproducibility, or denominator failure returns
`COUNTERFACTUAL_RESOLUTION_AUDIT_INVALID`. Stop immediately on failure.

## Frozen pathway analysis

The complete pathway panel is fixed, without outcome or intervention
selection, to `MAPK`, `PI3K`, `JAK-STAT`, `p53`, and `NFkB`. All 93
interventions are evaluated on all five axes. Drug-target annotations may only
be used after analysis for interpretation; they may not filter interventions,
define pathway eligibility, fit scores, or modify weights.

Use the existing signed PROGENy-compatible weights frozen in
`RESOLUTION2_PATHWAY_FREEZE.json`, map them to G_PRIMARY, fill absent genes with
zero, and verify unit L2 norm. No pathway predictor is trained. For each
pathway, project truth and replayed prediction with the same vector.

The primary pathway statistic pools the five coordinates in the original
paired replicate-stable ratio-of-sums recovery. Per-pathway recovery is
secondary. Ordinary Pearson correlation is not the primary metric.

## Frozen program analysis

Programs are secondary. Analyze only `K=4` and `K=8`. For every evaluation
episode, fit the response basis using only accessible training response rows
from both plates. Hidden target rows from both plates are excluded. Fit one
deterministic rank-8 thin SVD to training-row-mean-centered responses; K=4 is
its prefix. Resolve component signs by making the largest-absolute-loading
gene positive, with gene-axis order breaking ties. Project uncentered
displacements with the frozen basis. Global PCA is forbidden.

## Matched random controls

Follow `RESOLUTION2_NULL_PROTOCOL.md`. The pathway primary must exceed both the
gene reference and 500 reliability-matched signed random pathway-panel
projections. Each program K must be compared to 500 reliability-matched random
orthonormal subspaces. Matching can use evaluation truth after predictions are
frozen but cannot flow into prediction, basis selection, intervention
selection, or biological weight definition.

## Paired inference and multiplicity

Use 10,000 hierarchical paired bootstrap draws. Resample contexts, then
interventions within sampled contexts; keep plates and corresponding real/null
coordinates paired. Average frozen support-sequence contributions within target
entry before resampling. Use centered studentized max-absolute-deviation
simultaneous intervals.

- Primary pathway family: two budgets x pooled `{pathway-gene,
  pathway-random}` contrasts.
- Secondary pathway family: two budgets x five per-pathway contrasts, max-T.
- Program family: two budgets x two K values x `{program-gene,
  program-random}`, max-T.

No 360-family correction is introduced.

## Verdicts

`PATHWAY_RESOLUTION_POC_SUPPORTED` requires, at `(49,92)`, corrected positive
pooled pathway-minus-gene and pathway-minus-random intervals, a valid
replicate-stable denominator, and no leakage. The `(40,4)` estimates need only
be directionally consistent.

Program independently returns one of:

- `PROGRAM_RESOLUTION_POC_SUPPORTED`
- `PROGRAM_RESOLUTION_POC_PARTIAL`
- `PROGRAM_RESOLUTION_POC_NOT_SUPPORTED`

The overall verdict is one of:

- `COUNTERFACTUAL_RESOLUTION_PRINCIPLE_SUPPORTED`
- `COUNTERFACTUAL_RESOLUTION_PRINCIPLE_PARTIALLY_SUPPORTED`
- `COUNTERFACTUAL_RESOLUTION_PRINCIPLE_NOT_SUPPORTED`
- `COUNTERFACTUAL_RESOLUTION_AUDIT_INVALID`

Strong overall support requires the pathway primary and at least directional
program consistency or the already frozen Lea orthogonal observation. Partial
support covers program-only support or pathway-versus-gene gain without
pathway-versus-random separation. Not supported is available only after a
valid completed analysis. Invalid is reserved for replay, provenance, leakage,
or denominator failure.

Lea full-gene R2 approximately `0.003544` and PC1-4 R2 approximately `0.1787`
are orthogonal qualitative support only and are never numerically combined.

## Stop rule

After the frozen analysis, stop. Do not add pathways, lower null-matching
standards, change ontology, train pathway models, add architectures, select
pathways from outcomes, or implement an identifiable-gene head. Even a strong
result supports resolution-aware identifiability under the same prediction; it
does not establish that pathway-level Virtual Cells solve CGC.

