# CGC-EC-2 fixed metric definitions

Let `G=25,695`. Plate-specific predictions are fit independently. Dot products below are divided by `G`; ratios use unrounded sums.

## Per-entry full-response metrics

For each plate, target vector `y` and prediction `y_hat`:

- Pearson: gene-wise centered Pearson correlation.
- Cosine: `(y dot y_hat) / (||y|| ||y_hat||)`.
- Gene-space R2: `1 - sum((y-y_hat)^2) / sum((y-mean(y))^2)`.
- MSE: `mean((y-y_hat)^2)`.
- Variance retention: `var(y_hat) / var(y)`; it is not clamped.

Plate-specific values and their arithmetic plate mean are reported. Undefined zero-variance correlations remain missing rather than being set to zero.

## Context-specific excess response

For plate `r`, support set `S_m`, target `(c,p)`:

`B_r = mean_{s in S_m} Delta_r[p,s]`

`D_r = Delta_r[p,c] - B_r`

`Dhat_r = Deltahat_r - B_r`

`E_r = D_r - Dhat_r = Delta_r[p,c] - Deltahat_r`.

Thus M0 has `Dhat=0` and exactly `g=0` up to numerical precision.

## Replicate-stable recovery

Per episode:

`V_truth = (D_6 dot D_14)/G`

`V_after = (E_6 dot E_14)/G`.

For an analysis set `A`, including support trajectories with equal weight:

`g(A) = 1 - sum_A(V_after) / sum_A(V_truth)`.

This is a ratio of sums, not a mean of ratios, and is never clamped. Negative values indicate negative transfer. A grid point is invalid if its aggregate truth denominator is non-positive.

For full-response positive-control reporting, the no-response baseline is zero:

`g_full = 1 - sum((Delta_6-Deltahat_6) dot (Delta_14-Deltahat_14)) / sum(Delta_6 dot Delta_14)`.

This is never substituted for Target-B recovery.

## Confidence intervals

The fixed episode utility table is averaged over the eight support trajectories for each context-by-intervention entry. Each of 10,000 bootstrap draws:

1. samples 50 context indices with replacement;
2. for each sampled context, samples 93 intervention indices with replacement;
3. carries the matched Plate 6/14 `V_truth,V_after` pair together;
4. recomputes the ratio of sums without refitting.

Pointwise 95% intervals are the 2.5% and 97.5% bootstrap quantiles.

For the primary M2 Target-B grid, let `g_hat_j` be the estimate and `s_j` the bootstrap standard deviation. For each bootstrap draw `b`,

`T_b = max_j ((g_hat_j - g_bj) / max(s_j,1e-12))`.

With `q=.95 quantile(T)`, the simultaneous lower band is

`LCB_j = g_hat_j - q*s_j`.

The analogous upper band is retained for plots/null comparison. The family contains all 90 fixed `(m,k)` points; no point is dropped after inspection.

## Experimental-budget thresholds

`B(m,k)=93m+k`, `f(m,k)=B/4650`. The all-but-one check must equal `B(49,92)=4649`.

- `B_detect^95`: minimum observed budget with simultaneous `LCB>0`.
- `B25^95`: minimum observed budget with simultaneous `LCB>=0.25`.
- `B50^95`: minimum observed budget with simultaneous `LCB>=0.50`.
- `B80^95`: minimum observed budget with simultaneous `LCB>=0.80`.
- `B95^95`: minimum observed budget with simultaneous `LCB>=0.95`.

Ties are resolved by smaller `m`, then smaller `k`. If absent, the literal value is `NOT_REACHED_WITHIN_FULL_GRID`. Thresholds are computed for primary random-support M2. RNA-informed thresholds are separately labeled secondary.

## Null adjudication

At every grid point, each deterministic correspondence null produces its own paired `V_truth,V_after` table and bootstrap interval. `null_separated=true` only when the observed estimate exceeds the largest pointwise 95% upper bound across all applicable nulls. The simultaneous threshold remains numerically defined by the preregistered M2 band, but a threshold lacking null separation is not called correspondence-specific prediction.

## All-but-one summaries

At `(49,92)`, one unique prediction per `(context,intervention,plate,estimator)` is evaluated. Report global ratio-of-sums recovery, full-response and excess-response per-entry metrics, context-stratified ratio-of-sums, intervention-stratified ratio-of-sums, simultaneous confidence, and distributions. Support-seed duplication is prohibited.
