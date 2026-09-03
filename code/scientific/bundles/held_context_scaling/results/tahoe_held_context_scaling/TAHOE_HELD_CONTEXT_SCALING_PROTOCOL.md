# Tahoe heterogeneous held-context scaling protocol

## Scientific estimand

This frozen analysis asks whether increasing the number of heterogeneous response-observed cell-line contexts improves zero-shot recovery of an intervention-specific response operator in an entirely held-out Tahoe context. The only primary scaling variable is the number of response-observed training contexts, `m = 2, 4, 8, 12, 16, 24, 32, 40, 49`.

The manuscript-facing terms are **heterogeneous held-context scaling** and **zero-shot held-context scaling**. Tahoe is not described as a universal biological world model.

## Frozen source data

- Responses: existing `delta_primary` pseudobulk tensor from CGC Tahoe-0I, with Plate6 and Plate14 retained separately.
- Axes: 50 contexts, 93 exact shared drug-dose interventions, and the frozen 25,695-gene `G_PRIMARY` response universe.
- Primary response: treatment log1p(CPM) minus the equal-weight mean of two independently normalized DMSO-well log1p(CPM) profiles.
- Baseline features: the already audited control-only Tahoe-0D tensor containing four independently preserved DMSO samples (two wells on each plate), 50 contexts and 42,523 genes selected without treated outcomes. Equal-well Plate6 and Plate14 means are used separately.
- Baseline normalization for modeling: each frozen DMSO profile is centered across its genes and L2-normalized independently. This uses no other context and no perturbation outcome.
- No Tahoe individual-cell record is loaded or regenerated.

The 42,523-gene baseline feature axis is an outcome-blind input representation; every predicted response and every score remains on the fixed 25,695-gene response axis.

## Outer held-context episode

For target context `c*`, all 93 treated outcomes on both plates are sealed before fitting, tuning, centering and prediction. Only the two-well DMSO baseline is available at inference. The response-observed pool contains the other 49 contexts.

For each target and each of 20 deterministic independent ladders, one permutation of the 49-context pool is drawn and nested prefixes define all `m < 49` training sets. The unique `m = 49` set is evaluated once. The target, response gene axis and 93-intervention test universe are fixed across `m`.

The pilot targets were selected before outcome inspection by evenly spaced frozen context indices 0, 12, 24, 36 and 49.

## Models

1. Shared-response reference: the intervention-specific mean response over the current `m` training contexts; recovery is exactly zero by construction.
2. Linear context-kernel Ridge: a dual multi-output Ridge predictor whose context kernel is the inner product of baseline-only normalized DMSO profiles.
3. RBF context-kernel Ridge: the primary nonlinear predictor, with RBF distances defined only by baseline DMSO profiles.
4. Bilinear reduced-rank context-by-intervention model: a secondary model of the form `Delta[c,p,g] = mean[p,g] + z[c]^T B[p,g]`. Its response basis is fitted separately on each plate and only from the current training-context response tensor; final predictions are reconstructed on all 25,695 genes before Gram-equivalent evaluation.

Plate6 predictions use Plate6 baselines and Plate6 training responses. Plate14 predictions use Plate14 baselines and Plate14 training responses. Treated outcomes are never averaged across plates.

## Training-only model selection

For `m >= 8`, hyperparameters are selected by deterministic five-fold held-training-context validation, pooling all 93 interventions and 25,695 genes. The objective is the cross-plate prediction-error energy, so both response replicate directions inform selection while the outer target remains sealed. At `m = 2` and `m = 4`, nested selection is ill-posed; frozen defaults are used and these points are marked small-m descriptive sensitivities.

Linear Ridge candidates are `1e-4` through `1e3`. RBF Ridge candidates cross regularization values `1e-3` through `10` with bandwidth multipliers `0.25, 0.5, 1, 2, 4` around the training-set median-distance rule. No target response enters a kernel, bandwidth or regularization choice.

## Replicate-stable evaluation

For each intervention, Plate6 reference residual `r6`, Plate14 reference residual `r14`, and corresponding model errors `e6`, `e14` are evaluated through their full-gene cross products. Energies are summed over all 93 interventions before forming

`g = 1 - sum(<e6,e14>) / sum(<r6,r14>)`.

This is the existing audited Tahoe cross-plate energy construction. A dedicated numerical sanity case must exactly reconcile the generalized two-weight implementation with the frozen Tahoe `_residual_from_gram` helper and direct vector arithmetic.

The fixed-reference sensitivity holds the evaluation coordinate at the mean of all 49 non-target contexts. Responses outside `S_m` are used only for its denominator and are never exposed to model fitting or tuning.

The compatible cross-replicate decomposition is

- `alpha_cross = 0.5 * (<hatGamma6,Gamma14> + <Gamma6,hatGamma14>) / <Gamma6,Gamma14>`;
- `kappa_cross = <hatGamma6,hatGamma14> / <Gamma6,Gamma14>`;
- `cosine_cross = aligned / sqrt(truth_cross_energy * predicted_cross_energy)` when both cross energies are positive.

Under this exact aggregation, `g = 2 * alpha_cross - kappa_cross`; the equality is verified numerically rather than assumed from Fairfax.

## Aggregation and uncertainty

All 50 contexts are retained unless their prespecified reference cross-energy is non-positive, in which case the context ratio is undefined but its contribution remains visible. Pooled `g` is a ratio of pooled cross-energy sums. Median context `g` and the fraction of contexts with `g > 0` are also reported. Hierarchical bootstrap resamples contexts first and interventions within sampled contexts.

Empirical curves are written and hashed before any scaling law is fitted. Candidate laws for `q = 1 - g` are constant, continuing power, and power with residual floor; an exponential form is secondary. A finite asymptote is reported only if it is preferred, stable, bootstrap-informative and constrained by late-stage observations. Otherwise the verdict is `ASYMPTOTE_NOT_IDENTIFIED`.

## Secondary analyses

- Domain blocking uses the frozen Tahoe Organ/lineage labels. Prespecified adequate domains contain at least three contexts: Lung, Bowel, Pancreas, Skin and CNS/Brain. All contexts in the target domain are excluded from response-observed training.
- A semi-synthetic positive control encodes baseline-dependent susceptibility using real DMSO geometry, 93 interventions and two independent replicate directions. A baseline-independent context-effect null preserves response signal but breaks correspondence with baseline geometry.
- Fairfax remains a descriptive boundary only; its normalized single-measurement donor estimand is not pooled with Tahoe.

## State bridge boundary

The official CGC State ST-SE stability audit contains five held-out **Parse immune cell types** (`B_Intermediate_Memory`, `B_Naive`, `Plasmablast`, `CD4_Memory`, `CD14_Mono`), not five Tahoe cell-line contexts. No CVCL identifier or auditable cross-dataset mapping exists in the frozen State artifacts. Therefore the requested same-context Tahoe bridge is not scientifically defined. The bridge table will preserve the five official State rows and values, leave Tahoe trajectory fields missing, and explicitly report `NOT_DEFINED_NO_SHARED_CONTEXTS`; no biological identity will be invented.

## Stop rules and licensing

The full experiment may start only after the prescribed five-target, four-`m`, Ridge-plus-RBF pilot passes runtime, memory, disk, exact-`g` and leakage checks. Scientific output code added for this analysis is MIT-licensed. Source datasets and external model artifacts retain their original licenses. No manuscript, existing figure, Extended Data, supplement or author-formatted 600-dpi image is modified during adjudication.
