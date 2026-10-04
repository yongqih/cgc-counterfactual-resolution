# Fixed PROGENy pathway-resolution scaling protocol

Fixed before computing any full-panel subset outcome.

## Scope and authorities

- Scientific task: exhaustive fixed-library pathway-resolution scaling on the fixed Tahoe Experimental Compression primary all-but-one episode only.
- Resolution-2 provenance: `46b25eec4a97e3001cc43262980af8f69c533478`.
- Five-pathway fidelity provenance: `ab44618011cd963cadfcc0702d9c9ec71b87a8c3`.
- Exact external resource: local PROGENy package `1.17.3`, repository revision `cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f`.
- Exact mapped weight archive SHA-256: `f5c4faa94fe03b2d088b72743da175380dfcf2edbfacd0077cc25eb19d2ca7e9`.
- Exact 25,695-gene truth SHA-256: `005603c5a01c085762c9eecb266de6302f95356ca5eb4026f8c0380f57793f73`.
- Exact primary fixed prediction tensor SHA-256: `a1e99f8a949b8b205d1b360dd9e6919229e6b41969e98fdc57139f6983f798d6`.
- Exact all-but-one episode/source archive SHA-256: `b0e410e9c0f5d0d9f0a4d3fbfd02cf247d171cd203c480083dc6fa666b68d928`.
- Exact fixed full-gene utility archive SHA-256: `bd3b4f5097b4f158f58da102849d52080bb9ae7b7a4c4325f37bf574fd1a7c08`.
- Exact 10,000-draw hierarchical bootstrap weights SHA-256: `f0a9697175fe2d4b8bc800e4352d961ed349f9e9a33df138c793be24f982c9cf`.

No predictor is trained, refit, tuned or selected. No pathway definition is added, removed or updated after outcome inspection. Lea, WGCNA, Reactome, MSigDB and random projections are outside scope.

## Fixed pathway dictionary and exhaustive family

The complete panel in the exact fixed weight archive contains 14 PROGENy axes, in fixed source order:

`Androgen`, `EGFR`, `Estrogen`, `Hypoxia`, `JAK-STAT`, `MAPK`, `NFkB`, `PI3K`, `TGFb`, `TNFa`, `Trail`, `VEGF`, `WNT`, `p53`.

All `2^14-1=16,383` non-empty subsets are enumerated. The recorded panel is fixed as `MAPK`, `PI3K`, `JAK-STAT`, `p53`, `NFkB`. Subsets are never selected by Tahoe outcomes.

## Numerical span definition

For subset `S`, `W_S` is the exact `25,695 x |S|` mapped signed-weight matrix. Let `s_max(S)` be its largest singular value. A single relative cutoff is used for every subset:

`rcond = max(25,695, 14) * eps(float64) = 5.7054361235486795e-12`.

A singular direction is retained iff `s_i > s_max(S) * rcond`. The orthogonal projector is implemented from the retained eigensystem of `W_S^T W_S`, algebraically equivalent to `W_S(W_S^TW_S)^+W_S^T`; no gene-by-gene projector is materialized. Rank, singular values, absolute tolerance, `cond(W_S)` and `cond(W_S^TW_S)` are recorded for every subset.

## Fixed episode and metrics

Only `m=49,k=92` is used. For each plate `p`, target context `c` and intervention `i`,

`D[p,c,i] = X[p,c,i] - mean_{s != c} X[p,s,i] = (50/49)(X[p,c,i]-mean_c X[p,c,i])`.

Plate 6 and Plate 14 remain separate until the cross product.

For the orthogonal span projector `P_S`, biological fidelity is

`F(S) = sum_ci D6_ci^T P_S D14_ci / sum_ci D6_ci^T D14_ci`.

The fixed Experimental Compression prediction is `Dhat`. Recorded pathway recoverability uses the raw unit-norm PROGENy score coordinates, not a whitened orthogonal basis:

`z_p = W_S^T D_p`, `e_p = W_S^T(D_p-Dhat_p)`,

`g(S) = 1 - sum_ci z6_ci^T z14_ci / denominator` is **not** the definition. The exact definition is

`g(S) = 1 - sum_ci e6_ci^T e14_ci / sum_ci z6_ci^T z14_ci`.

The implementation reconstructs `Dhat` from the exact fixed all-but-one source and weight archive in float64, as in the five-pathway calculation, and independently checks its pathway scores against the stored float32 prediction tensor.

## Anchor gate

Before enumeration, require:

- mapped recorded weight vectors equal the fixed five-pathway loader exactly;
- `|F5 - 0.029695165| <= 5e-7`;
- `|g5 - 0.27281393788398667| <= 1e-10`;
- `|g_gene - 0.11068053088808294| <= 1e-10`;
- `|(g5-g_gene) - 0.16213340699590373| <= 1e-10`;
- the reconstructed full-gene truth table matches the fixed table within `5e-7` per context-intervention entry;
- all input hashes, axes and all-but-one sources pass.

Failure is fail-closed and yields `PATHWAY_RESOLUTION_SCALING_INVALID`; no subset outcome is generated.

## Metric reconciliation

`F(S)` uses the orthogonal quadratic form `P_S=W_S(W_S^TW_S)^+W_S^T`. Recorded `g(S)` uses the raw-score quadratic form `W_SW_S^T`. Because the fixed PROGENy vectors are non-orthogonal, these are not the same numerator/denominator measure for multi-pathway subsets. Therefore `F(S)g(S)` is not algebraically a fraction of full-gene reproducible signal recovered by the prediction. `H` is not computed or plotted. Singleton equality is a special case and does not make a cross-resolution product valid.

## Complete composition summaries and biological bootstrap

For every pathway count and effective rank, summarize the complete subset distributions of `F` and `g` by median, IQR, 10th-90th percentiles, minimum and maximum. These envelopes quantify pathway-composition dependence and are not confidence intervals.

Biological uncertainty reuses all 10,000 existing context-then-intervention hierarchical bootstrap draws. At the prespecified grid `k={1,2,3,5,8,11,14}`, each draw recomputes every subset metric at that `k` and then takes the median across subsets. Subsets are never resampled or treated as biological replicates. Report percentile 95% intervals and bootstrap standard errors.

## Prespecified descriptive relationships

Across the 14 complete-curve medians, fit without model selection claims:

1. log-linear: `y=a+b log(k)`;
2. saturating exponential: `y=a+b(1-exp(-lambda k))`, with `lambda` searched on a fixed 400-point log grid from `1e-3` to `3` and `a,b` solved by least squares;
3. positive power-law-like: `y=a k^b`, only when every fitted median is positive.

Report R-squared, RMSE, lag-1 residual correlation and bootstrap parameter intervals on the prespecified seven-point bootstrap grid. A higher R-squared alone does not choose a model.

## Fixed adjudication rules

The primary fidelity trend passes when all hold:

- Spearman correlation between rank and the 14 median `F` values is at least `0.90`;
- at least `97.5%` of bootstrap draws have positive seven-grid Spearman correlation;
- the 95% bootstrap interval for median `F_14 - F_1` is wholly above zero;
- median adjacent-grid stochastic-dominance probability for `F` is at least `0.75`, with no adjacent probability below `0.60`.

Recoverability is systematic if either (a) `|rho(rank, median g)| >= 0.75` and at least `97.5%` of bootstrap correlations share its sign, or (b) it is stably flat, defined before outcomes as `|median g_14-median g_1| <= 0.05` with its 95% bootstrap interval contained in `[-0.10,0.10]`.

Composition resistance requires resolution to explain at least 50% of complete-subset `F` variation (`eta^2 >= 0.50`) and, for directional `g`, at least 20% of `g` variation; for stably flat `g`, the median within-count 10th-90th width must be at most `0.15`.

- `PATHWAY_RESOLUTION_SCALING_SUPPORTED`: primary fidelity, systematic recoverability and composition-resistance rules all pass.
- `PATHWAY_RESOLUTION_SCALING_PARTIAL`: the primary fidelity trend passes, but at least one recoverability/composition-resistance rule fails.
- `PATHWAY_RESOLUTION_SCALING_NOT_SUPPORTED`: the primary fidelity trend fails without an implementation defect.
- `PATHWAY_RESOLUTION_SCALING_INVALID`: an input, anchor, fairness, leakage or numerical gate fails.

The phrase **scaling law** is allowed only if the overall verdict is supported and one prespecified functional family has `R^2 >= 0.95` for both `F` and `g`, stable parameter direction in at least 97.5% of bootstrap draws, and no obvious residual trend (`|lag-1 residual correlation| < 0.50`) for both metrics. Otherwise the result is a pathway-resolution scaling curve/relation.

The recorded five-pathway set is marked and its empirical within-`k=5` percentiles and normalized distance to the Pareto frontier are reported. “Near frontier” is descriptive only and is defined as normalized Euclidean distance at most `0.05` in the observed `F x g` rectangle.

The full-panel fidelity is called substantial only at `F >= 0.25`; below this threshold it remains a coarse, incomplete representation. This threshold affects wording, not the scaling verdict.
