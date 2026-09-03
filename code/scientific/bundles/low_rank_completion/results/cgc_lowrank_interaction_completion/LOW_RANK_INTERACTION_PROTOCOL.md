# Tahoe low-rank interaction completion — frozen protocol

Status: **FROZEN BEFORE LOW-RANK HELD-OUT OUTCOME INSPECTION**

Frozen source authority: `codex/cgc_entrywise_experimental_compression` at
`a893cc338c11f402a9ef9966bd376f2aca92cb88`. The historical strict all-but-one
M2 result (`g=0.11068053088808294`) is context only and is not the matched
comparator if the fallback design is used.

## Scientific target

The response for context `c`, intervention `p`, and gene `g` is modeled as

`Y[c,p,g] = mu[g] + A[c,g] + B[p,g] + sum_r U[c,r] V[p,r] Q[r,g] + error`.

The primary endpoint is the frozen context-specific cross-plate recovery

`g = 1 - sum_i <D6_i-Dhat6_i, D14_i-Dhat14_i> / sum_i <D6_i,D14_i>`,

where the reference response for a target entry is the equal-weight mean of
the other 49 contexts for the same intervention. Full-response recovery uses
the identical residual numerator and the uncentered response denominator.

Plate 6 and Plate 14 are fit independently. They share masks only. They are
combined only in the frozen cross-plate estimand.

## Input freeze

- 50 contexts, 93 exact drug-by-dose interventions, 4,650 entries.
- `G_PRIMARY`, 25,695 genes.
- `delta_primary`: treatment `log1p(CPM)` minus the equal-weight mean of the
  two independently normalized DMSO-well profiles.
- Frozen sample order: context-major, intervention-minor.
- Same-plate and cross-plate Gram caches and the materialized `G_PRIMARY`
  response tensor must match the hashes recorded by the run audit.
- No external annotation, baseline RNA, drug descriptor, pathway, or target
  label is supplied to either comparator.

## Hide-first contract

An outer-fold target set is removed before construction of the same-plate
training Gram, additive design fit, inner mask, factor fit, hyperparameter
selection, affine weight fit, or prediction. Training APIs receive only the
observed-row Gram. Prediction is stored as coefficients on observed response
rows. Any mutation confined to outer targets must leave the masks,
hyperparameters, factors, coefficients, and materialized predictions exactly
unchanged. Failure is `LOW_RANK_INTERACTION_COMPLETION_INVALID`.

## Executability gate

Before any low-rank target outcome is evaluated, four target episodes are
chosen by SHA256 ordering of the frozen `(context_id, intervention_id)` axis at
the 0%, 33%, 67%, and 100% order statistics. Both plates, every rank/ridge
candidate, both frozen starts, train-side validation, and the final refit are
timed without scoring the outer targets.

Exact 4,650-fit all-but-one is executable only if the extrapolated total wall
time is at most 24 GPU-hours, peak allocated VRAM is at most 10.5 GiB, and at
least 95% of representative candidate fits meet the frozen convergence gate.
Otherwise the fallback below is mandatory. This decision cannot use outer
target performance.

## Frozen fallback masks

Fallback uses 100 balanced entry folds. Context and intervention axes are
independently permuted with seed `202608251`. If `a(c)` and `b(p)` are their
permuted ranks, the outer fold is

`fold(c,p) = (b(p) + 2*a(c)) mod 100`.

Every entry is predicted exactly once. Every fold contains 46 or 47 targets,
at most one target per context, and at most one target per intervention. Thus
each fit observes 4,603 or 4,604 entries (98.9892% or 98.9677%; exactly 99.0%
when pooled across predictions).

An independent axis permutation with seed `202608252` defines an analogous
100-color inner assignment. For outer fold `f`, the inner validation color is
`(37*f + 11) mod 100`; overlap with the outer target set is removed. The
remaining outer-observed entries are the inner-training set. Masks are
identical across plates.

## Low-rank fit

The additive design is an intercept plus treatment-coded context and
intervention main effects (142 full-rank columns). For fixed CP entry scores
`Z[c,p,r]=U[c,r]V[p,r]`, additive effects are removed exactly within the
observed set by Frisch--Waugh--Lovell projection. Gene loadings `Q` are
profiled analytically from the observed-row Gram; no response basis is learned
from a hidden row. Context and intervention factors are zero-centered.

The loss is observed-entry squared reconstruction error with L2 penalties on
the profiled gene loadings and balanced latent factors. Component gauge is
balanced after each step without changing `U[c,r]V[p,r]`. Optimization uses
deterministic CUDA Adam, learning rate `0.03`, maximum 400 steps, minimum 80
steps, a 30-step relative objective tolerance of `1e-3`, or global gradient
norm at most `1e-4`. CUDA deterministic algorithms are required.

Ranks are exactly `[2,4,8,16,32]`. Dimensionless L2 values are exactly
`[0.001,0.01,0.1]`, applied as `lambda * n_observed` to the profiled loading
normal matrix and as `lambda` to mean squared context/intervention factors.
Each candidate has two fixed starts. The lower train-objective start is kept.
The candidate with the lowest same-plate full-response inner-validation SSE is
selected; numerical ties within relative `1e-9` prefer smaller rank and then
larger L2. Final factors are refit on all outer-observed entries with the same
two starts. Selection and fitting are independent by plate and outer fold.

The additive comparator is the model's `mu+A+B` component before adding its CP
interaction. The primary increment is
`Delta g_interaction = g_full_lowrank - g_additive`.

## Matched affine comparator

The frozen M2 affine context-weight estimator and its exact sum-to-one
constraint are rerun on each fallback outer mask. The target intervention uses
the other 49 contexts. All 92 target-context sentinels are retained. If a
non-target source-sentinel entry is also absent in the same outer fold, it is
replaced only by the outer-train additive prediction; target outcomes are never
imputed into training. Ridge values are the target-context/plate-specific
`m=49,k=92` values frozen by the earlier reference-only pseudo-target CV. This
is conservative prior tuning for the affine comparator and is not reselected
from the new held-out outcomes. On a one-entry mask this implementation must
reproduce the original M2 prediction coefficients within `1e-8`.

Both models receive the exact same observed entries. The affine model's fixed
algorithm uses a target-specific subset of that information; the tensor model
uses all observed entries because that is its declared completion structure.

## Inference

- Use the existing 10,000 frozen hierarchical context-then-intervention
  bootstrap count table, preserving plate pairing.
- Report point estimate and percentile 95% interval for each model and paired
  difference. No refit occurs inside the bootstrap.
- Context summaries pool the 93 entries within each context.
- Diagnostics use only pre-existing quantities: symmetric truth-aligned
  amplitude, cosine/alignment, predicted-to-true scale ratio, and CP
  interaction-energy fraction.

## Synthetic implementation control

Seed `202608253` generates a 50x93x512 rank-4 CP interaction with independent
plate noise and the exact frozen outer/inner masks. Signal and noise are fixed
so replicate-stable interaction reliability is 0.90 in expectation. The
implementation control passes only if full low-rank context-specific recovery
is at least 0.80, its gain over additive is at least 0.50, every entry is OOF,
and at least 95% of selected final fits converge. It cannot tune the real-data
model.

## Validity and verdict gates

`LOW_RANK_INTERACTION_COMPLETION_INVALID` is returned for leakage/invariance,
axis/hash, mask/fairness, synthetic-control, nonfinite, or selected-final-fit
convergence failures (fewer than 95% converged).

Among valid runs:

- `LOW_RANK_INTERACTION_COMPLETION_CLOSES_GAP`: paired low-rank-minus-affine
  95% lower bound above zero, point gain at least 0.25, and low-rank primary
  `g >= 0.50`.
- `LOW_RANK_INTERACTION_COMPLETION_PARTIAL`: paired lower bound above zero and
  point gain at least 0.05, without meeting the close-gap rule.
- `LOW_RANK_INTERACTION_COMPLETION_NO_RESCUE`: every other valid result.

These thresholds are frozen before real low-rank held-out outcomes are
computed. No rank, L2, start, mask, metric, or verdict threshold may be changed
after inspection.
