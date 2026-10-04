# Lea RNA-to-response semi-synthetic identifiability positive control

Status: **FROZEN BEFORE SEMI-SYNTHETIC OUTCOME INSPECTION**

Identity: `SEMI_SYNTHETIC_BASELINE_ENCODED_RESPONSE_CONTROL`

## Scientific scope

This positive control tests whether the exact held-LCL Lea pipeline can detect a matched-dimensional response when transferable individual-specific response structure is deliberately encoded in the observed baseline transcriptome. It is a pipeline-sensitivity and positive-control recoverability experiment, not an estimate or upper bound on the information content of real RNA.

## Fixed real-data provenance

- Provenance branch: `codex/level2_lea_rna_counterfactual`.
- Provenance HEAD: `4424e4acbbcc593f61b7556282f45f12e29802fc`.
- Fixed full-battery result commit: `c3d074412ecc1be0958141726db16a1ed0036a04`.
- Biological units: the exact 342 paired LCLs.
- Expression rows: the exact 10,157-row post-SVA processed representation.
- Predictor: the exact fixed ETOH baseline matrix in `LEVEL2_OOF_FROZEN.npz`.
- Splits: the exact fixed five biological-LCL-disjoint outer folds.
- Metric: pooled full-gene OOF residual R2, using each fold's outer-training response mean exactly as in the real analysis.

Before any synthetic fit, reconciliation must reproduce the six fixed real-data values recorded in `configs/lea_semisynthetic_control_frozen.json`. Failure returns `LEA_FROZEN_PIPELINE_MISMATCH` and stops.

## Fixed generator

Primary latent rank is `r=4`. The optional `r=8` secondary is disabled and will not be run.

For each of the ten fixed seeds `207049` through `207058`:

1. Standardize every real baseline-expression row across the 342 LCLs using the same mean/scale convention as the Lea pipeline (`scale<1e-6` becomes one).
2. Generate a Gaussian `10157 x 4` matrix and take its deterministic reduced QR factor as the orthonormal baseline projection `W_X`.
3. Set `z_X=X_standardized W_X`, then standardize each of its four coordinates across all 342 LCLs to mean zero and population variance one. This global operation defines synthetic truth from observed baseline only; it does not fit a predictor or access a response outcome.
4. Generate independent `z_H ~ N(0,I_4)`.
5. Generate a second Gaussian `10157 x 4` matrix, take its reduced QR factor, and transpose it to obtain a four-row orthonormal response loading `W_Y`.
6. For the fixed grid `lambda=[0,0.05,0.10,0.25,0.50,0.75,1.00]`, construct `z(lambda)=sqrt(lambda)z_X+sqrt(1-lambda)z_H` and `Gamma_signal=z(lambda)W_Y`.
7. Generate two independent Gaussian measurement-error matrices. Their variance is initialized analytically as `var(signal)*(1-0.7443)/0.7443`. The realized flattened Pearson correlation of replicates A and B must be within 0.015 of 0.7443 for every seed/lambda or the run stops with `NOISE_CALIBRATION_FAIL`.

Replicate A is the only prediction target. Replicate B is used only for reliability validation and cannot enter features, training, model selection, or hyperparameters. `lambda` is solely the deliberately baseline-dependent fraction of latent variance in this generator; it is not mutual information or a fraction of real biological information.

Random streams for `W_X`, `z_H`, `W_Y`, replicate-A noise, and replicate-B noise are deterministic, disjoint, and recorded by seed and lambda index.

## Fixed estimators and preprocessing

The exact existing Lea implementations are reused:

- dual Ridge;
- fold-local PCA plus Ridge;
- pilot two-hidden-layer MLP;
- exact RBF kernel Ridge;
- PCA-program HistGradientBoosting decoder;
- five-block deep residual MLP.

No inner tuning is performed on synthetic outcomes. Every outer fold reuses the exact real-data-selected Ridge alpha, PCA dimension/alpha, pilot-MLP epoch count, RBF gamma multiplier/alpha, boosting setting, deep setting, deep epoch count, and three-seed prediction averaging. Baseline standardization and PCA remain fold-local. The test LCLs never enter model fitting.

## Outcome-blind runtime design

The preselected design is the tiered design:

- Tier 1: Ridge, RBF kernel Ridge, and deep residual MLP at all seven lambda values and ten seeds.
- Tier 2: all six models at lambda `0`, `0.25`, and `1` for all ten seeds.
- Tier-1 results at the three Tier-2 lambda values are reused, never recomputed.

This selection was made before synthetic outcome inspection because the full design would repeat high-dimensional neural and boosting fits at intermediate calibration points that are unnecessary for the six-model endpoint confirmation. A response-free matched-shape timing benchmark may estimate runtime but cannot produce or inspect a scientific R2.

All formal computations are fold-resumable and atomic. A completed cache with matching generator/config/source hashes is reused; a mismatched cache is rejected rather than silently overwritten.

## Primary endpoint and fixed adjudication

The primary endpoint is pooled full-gene OOF residual R2 for every executed `(model,lambda,seed)`.

`PIPELINE_POSITIVE_CONTROL_PASS` requires all four:

1. at lambda zero, the median across seeds of the seedwise best R2 across the six preregistered models is at most 0.02;
2. at lambda one, at least one preregistered Tier-1 model has median R2 at least 0.25;
3. for that same qualifying model, the median paired lambda-one minus lambda-zero difference is at least 0.20 and all ten paired differences are positive;
4. across all seven lambda values, the Spearman correlation between lambda and that model's median R2 is at least 0.90.

If no model passes all four gates but at least one executed model has median lambda-one R2 above its median lambda-zero R2, return `PIPELINE_POSITIVE_CONTROL_PARTIAL`. Otherwise return `PIPELINE_POSITIVE_CONTROL_FAIL`.

The qualifying model may be any of the three preregistered Tier-1 models; it is not substituted into the real-data analysis.


## Interpretation

A pass would show that the near-zero real full-gene result is not forced by sample size, output dimensionality, or evaluation code alone. It would not estimate real RNA information, prove an information-theoretic limit, or identify a synthetic lambda corresponding to the real result.
