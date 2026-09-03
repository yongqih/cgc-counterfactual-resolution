# CGC-EC-2 strict prospective leave-one-entry-out experimental compression

Status: **FROZEN BEFORE NEW OUTCOME INSPECTION**

## Scientific estimand

For every Tahoe perturbation-matrix entry `(target_context, target_intervention)`, the target response is sealed before any fit-side object is constructed. The experiment measures how the number of observed perturbation entries

`B(m,k) = 93*m + k`, with `B_full = 50*93 = 4650`,

changes prospective recovery of the sealed response. DMSO controls are assumed observed in every context and are not counted among the 4,650 perturbation entries.

This is a new experiment. Historical CGC-0J remains `TRANSDUCTIVE_TWO_WAY_CENTERED_ANATOMY`; no CGC-0J target, cache, result, or verdict is overwritten or reinterpreted as prospective.

## Frozen data axis

- Study: Tahoe-100M frozen replicate core.
- Contexts: 50.
- Exact drug-by-dose interventions: 93.
- Plates: Plate 6 and Plate 14, kept separate for fitting and paired only during evaluation.
- Response: `delta_primary`, treatment `log1p(CPM)` minus the equal-weight mean of the two independently normalized DMSO-well `log1p(CPM)` profiles.
- Gene universe: `G_PRIMARY`, exactly 25,695 coverage-only genes; gene hash is recorded in `ENTRYWISE_INFORMATION_SET.json`.
- No raw H5AD is reprocessed. The audited CGC-0I response Zarr is read-only.
- Baseline RNA for the secondary analysis is the four-well control-state artifact; treated outcomes never select RNA neighbors.

## Hide-first dependency rule

For an episode `(c*,p*)`, the response rows `(plate6,c*,p*)` and `(plate14,c*,p*)` are replaced by an inaccessible sealed token in every fit-side view before normalization, Gram selection, support construction, hyperparameter selection, basis construction, coefficient fitting, or prediction. Evaluation truth is held in a separate evaluation object that model code cannot access.

Any cached global sufficient statistic is an input acceleration artifact, not a fit-side object. Episode access is mediated by a sealed-entry view that masks every row/column influenced by `(c*,p*)` before exposing a statistic to an estimator. The hard invariance test replaces the sealed response with four adversarial alternatives and requires bitwise-identical support identities, hyperparameters, weights/coefficients, and predictions. Failure returns `ENTRYWISE_PROSPECTIVE_INTEGRITY_FAIL` and stops all biological interpretation.

No full-matrix two-way-centering, target-context intervention mean, target truth normalization, outcome-balanced support selection, or target-truth model selection is permitted.

## Frozen experimental-support grid

- Reference contexts: `m = [1,2,4,8,16,24,32,40,49]`.
- Target-context sentinels: `k = [0,1,2,4,8,16,32,64,80,92]`.
- Primary support: eight fixed, nested, outcome-independent context trajectories, reusing sequences 0--7 from the frozen CGC-SUPPORT-0C manifest.
- Primary sentinels: eight fixed, nested, outcome-independent intervention permutations. For target `p*`, the first `k` identities after filtering out `p*` are observed.
- The paired context/sentinel trajectory index is the support-seed unit. All 50 contexts and all 93 interventions are evaluated.
- At `(m=49,k=92)`, all support seeds reduce to the same all-but-one information set; it is evaluated once per target entry.

All orders, seeds, inner folds, and correspondence-null maps are recorded in `ENTRYWISE_SPLIT_MANIFEST.json`.

## Estimators

`M0_SUPPORT_MEAN` predicts the mean response to `p*` over the `m` observed reference contexts.

`M1_SENTINEL_OFFSET` adds a target-generic sentinel offset to M0. The offset is the mean, across the `k` sentinels, of target response minus reference-context support mean. Its shrinkage coefficient is selected from `[0, 0.125, 0.25, 0.5, 0.75, 1]` by reference-context pseudo-target cross-validation. At `k=0`, the coefficient is zero.

`M2_AFFINE_RIDGE` is primary. It fits context weights on target sentinels with the exact affine constraint `sum(w)=1` and ridge penalty selected from `[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000]`.

`M3_LOWRANK_CONTEXT` restricts affine context weights to a reference-only context covariance eigenspace. Candidate ranks are `[1,2,4,8,16,24,32]`, capped at `m-1`; rank is selected by the same reference-only pseudo-target procedure. It never factors the completed 50-by-93 target matrix.

For M1--M3, hyperparameters are selected separately by plate, target context, `m`, and `k`, pooling the eight frozen support trajectories. Selection uses only the 49 non-target reference contexts. Five frozen intervention folds provide inner train/validation partitions. For each fold, target-budget-matched sentinel identities are drawn from the training folds; validation is on the held fold. Up to five deterministic reference contexts are pseudo-targets. No response from `c*` enters hyperparameter selection. Degenerate cases (`m=1`, insufficient rank, or `k=0`) use the predeclared M0/equal-weight limit.

The operational comparison may report a reference-CV-selected estimator, but all primary thresholds are defined on `M2_AFFINE_RIDGE`; estimator choice cannot be made from sealed-entry outcomes.

## Primary and secondary targets

Target A is the full response vector `Delta[p*,c*]`.

Target B is primary biologically. For each plate and episode,

`B[p*,c*] = mean_{c in S_m} Delta[p*,c]`

and

`D[p*,c*] = Delta[p*,c*] - B[p*,c*]`.

The predicted excess is `D_hat = Delta_hat - B`. All quantities use only the episode's observed reference contexts.

## Inference and thresholds

- Bootstrap draws: 10,000.
- Bootstrap seed: 202608225.
- Unit preservation: resample contexts, then interventions within resampled contexts; Plate 6/14 pairing is never broken. Frozen support trajectories are averaged within entry before resampling.
- Pointwise intervals: percentile 95% intervals.
- Primary simultaneous band: studentized bootstrap max-T lower band across all 90 frozen `(m,k)` points for M2 Target B.
- Thresholds: `tau = [0,0.25,0.50,0.80,0.95]`; the minimum observed `B(m,k)` whose simultaneous lower band reaches `tau`. No interpolation or extrapolation. Missing thresholds are `NOT_REACHED_WITHIN_FULL_GRID`.
- The raw discrete surface and Pareto non-dominated frontier are both retained; no monotonic smoothing is used for threshold claims.

## Positive controls and nulls

Positive control A requires M0 at `(49,0)` to recover a positive cross-plate-stable component of the full response with bootstrap lower bound above zero and above its intervention-identity null.

Positive control B is a 50-context, 93-intervention, two-plate synthetic matrix with a known rank-4 context operator, 512 genes, observed-scale shared/context/noise components, and fixed seed 202608224. It must show increasing M2 recovery along `(1,0) -> (4,4) -> (8,8) -> (16,16) -> (32,32) -> (49,92)` and cross the 25%, 50%, and 80% recovery levels at least once. Failure prevents power-based interpretation of real-data non-recovery.

Positive control C is the explicitly illegal-deployment `REPLICATE_ORACLE_CEILING`: one plate's sealed entry predicts the other plate. It calibrates assay noise only and is never an eligible estimator.

Four correspondence nulls are frozen for the full grid: intervention identity, context identity, sentinel identity, and reference-context identity. Each uses deterministic derangements in the split manifest. A positive biological claim must report whether the corresponding observed configuration exceeds every null; a confidence interval above zero alone is not described as correspondence-specific success.

## Secondary experimental design

Only after the invariance and positive-control gates pass and the primary random-support surface is frozen:

1. `RNA_NEAREST`: rank reference contexts by Euclidean distance in a rank-32 SVD of baseline control RNA. Gene centering/scaling and SVD use baseline controls only.
2. `REFERENCE_DOPT_SENTINELS`: greedy D-optimal/diversity selection using reference-context responses and target baseline RNA only.

RNA-informed thresholds are recomputed with the same estimators, metrics, bootstrap, and simultaneous-band definition. No treated target outcome selects a context or sentinel.

## Frozen stopping and validity rules

- Any hidden-target invariance failure: stop with `STRICT_ENTRYWISE_EXPERIMENTAL_COMPRESSION_INVALID`.
- Any axis, source hash, plate pairing, budget, or all-but-one accounting failure: stop invalid.
- Positive-control failure does not alter the real-data numbers, but blocks an interpretable negative conclusion and is prominently reported.
- The final report first returns the integrity label, then quantitative results. No estimator, grid, seed, null, threshold, or interpretation is changed after outcome inspection.

