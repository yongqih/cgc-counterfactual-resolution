# CGC-EC-M1 Secondary Frozen-Prediction Adjudication

## Identity and claim boundary

This is a `POST_HOC_SECONDARY_FROZEN_PREDICTION_ADJUDICATION`. It was
specified after observing that frozen estimator M1 had higher descriptive
recovery than preregistered primary M2. It does not replace, revise, or rescue
the CGC-EC-2 primary result.

The sole question is whether the already-frozen sentinel-offset estimator is a
reproducible partial remedy for context geometry compression.

## Frozen inputs

- Audit base: `a893cc338c11f402a9ef9966bd376f2aca92cb88`.
- Upstream frozen protocol commit:
  `555d3f67cb6d7d71ea293cc94dd7151562599492`.
- Upstream result commit:
  `dad1ca9dc86a6e0104976026810a246c20feba52`.
- Matrix: 50 contexts x 93 interventions x two plates x 25,695 G_PRIMARY
  genes.
- Grid: `m=[1,2,4,8,16,24,32,40,49]` and
  `k=[0,1,2,4,8,16,32,64,80,92]`.
- Eight frozen nested support/sentinel paths; the all-but-one corner uses its
  single frozen path exactly as in CGC-EC-2.
- M0/M1/M2/M3 residual sufficient statistics are read directly from the
  immutable upstream target caches and aggregate utility cache.
- Frozen alpha, support sets, sentinel sets, split maps, seeds, gene universe,
  and bootstrap units are unchanged.

No estimator is refit, retuned, replaced, or selected. Because M1 null residual
sufficient statistics were not stored upstream, the four null predictions may
be deterministically re-evaluated from the immutable Gram matrices, frozen M1
alpha values, and frozen information sets. This operation contains no fitting,
model selection, or outcome-dependent parameter choice. If those inputs fail
hash/schema/integrity checks, the audit stops as
`M1_SECONDARY_AUDIT_NOT_EXECUTABLE_FROM_FROZEN_PREDICTIONS`.

## Estimands

For each estimator `e`, grid cell, context, and intervention, let `V` be the
frozen cross-plate context-specific truth energy and `R_e` the frozen residual
energy. Recovery is

`g_e = 1 - sum(R_e) / sum(V)`.

The paired contrast is computed on the identical episodes:

`Delta_1-2 = g_M1 - g_M2`.

The primary secondary endpoint is all-but-one (`m=49,k=92`, budget 4,649).
The secondary endpoint is the complete frozen grid.

## Bootstrap and multiplicity

- Exactly 10,000 hierarchical paired bootstrap draws.
- Seed: `202608225`, identical to CGC-EC-2.
- Resample 50 contexts with replacement and, within each sampled context,
  resample 93 interventions with replacement.
- Pointwise intervals are percentile 2.5%/97.5% intervals.
- The estimator-family band uses the original studentized maximum-deviation
  construction jointly over 4 estimators x 90 grid cells (family size 360).
- A separate paired M1-minus-M2 studentized maximum-deviation band covers all
  90 frozen grid cells.
- All-but-one inference is reported from the corresponding members of these
  predeclared families; no point is selected after results.

## Frozen correspondence nulls

The names, maps, and semantics remain those frozen in CGC-EC-2:

1. intervention identity;
2. context identity;
3. sentinel identity;
4. reference context.

M1 null predictions use the frozen alpha and no refitting. The reference null
is the frozen cyclic shift of source-context weights. M1 uses equal reference
weights, so invariance of M1 to this operation is retained as an audit result,
not repaired by inventing a different null.

For every null, report observed-minus-null recovery and a paired one-sided
bootstrap p-value `(1 + count(Delta_draw <= 0)) / 10001`. Multiplicity across
the 4 nulls x 90 grid cells uses single-step max-statistic adjustment of the
paired bootstrap differences. All-but-one must pass every frozen null for the
strong confirmation verdict.

## Component decomposition

For each legal hide-first episode, define the unshrunk context-wide component
from legal sentinels only:

`A_c = mean_{j in K, j != p*} D_{j,c}`

and `I_pc = D_pc - A_c`. The M1 prediction component is `alpha A_c`; its
intervention-specific prediction is identically zero. Cross-plate sufficient
statistics quantify `V(A)`, `V(I)`, their two cross terms, reconstruction of
`V(total)`, the M1 recovery on `A`, and the residual recovery on `I`. For `k=0`,
the component decomposition is undefined and is reported as such. No hidden
target response contributes to `A_c`.

## Scaling, heterogeneity, and robustness

- Sentinel scaling: M1 and M1-minus-M2 across the frozen k grid at fixed m;
  report successive marginal gain, Spearman trend, and the first k reaching
  95% of the within-m descriptive maximum (descriptive saturation).
- Reference scaling: analogous summaries across m at fixed k.
- Heterogeneity uses all-but-one paired episode sufficient statistics. Context
  and intervention recovery use the same ratio-of-sums definition; positivity,
  median, IQR, and concentration are reported without external annotations.
- Cross-plate robustness uses the frozen all-but-one same-plate metrics for
  Plate6 and Plate14 and the cross-product pooled utility. It is descriptive
  for same-plate directions and inferential for the pooled cross-product.
- Full-response versus context-specific response is reported at all-but-one
  from frozen M1 sufficient statistics.

## Thresholds

From the 360-member estimator-family simultaneous lower bound for M1, report
the minimum budget attaining detection and 25%, 50%, 80%, and 95% recovery.
No interpolation or extrapolation is allowed. A point estimate above 25% with
lower bound below 25% is labelled
`DESCRIPTIVE_RECOVERY_ABOVE_25_PERCENT_BUT_25_PERCENT_NOT_SIMULTANEOUSLY_CONFIRMED`.

## Verdict rules

The terminal label is exactly one of:

- `M1_SENTINEL_CALIBRATION_PARTIAL_REMEDY_CONFIRMED`: all-but-one M1 lower
  bound is positive under the 360-member band; all four correspondence nulls
  pass; the 90-member paired M1-minus-M2 lower bound is positive at
  all-but-one; and positive recovery occurs in a substantial majority (more
  than half) of both contexts and interventions.
- `M1_SENTINEL_CALIBRATION_SIGNAL_SUPPORTED_BUT_MAGNITUDE_UNCERTAIN`: positive
  and null-separated signal is robust but the 25% magnitude is not
  simultaneously confirmed.
- `M1_DESCRIPTIVE_ADVANTAGE_NOT_CONFIRMATORY`: estimator/grid correction does
  not survive or any required correspondence null fails.
- `M1_EFFECT_HETEROGENEOUS_SUBSET_SPECIFIC`: pooled signal survives but 50% or
  fewer contexts or interventions have positive recovery.

The null-failure rule takes precedence over the heterogeneity and
magnitude-uncertain labels. The strong label never overrides the original M2
primary designation.

