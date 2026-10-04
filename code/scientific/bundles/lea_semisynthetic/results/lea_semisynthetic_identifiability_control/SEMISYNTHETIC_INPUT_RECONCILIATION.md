# Lea semi-synthetic input reconciliation

Status: **LEA_FROZEN_PIPELINE_MATCH**

- Provenance branch: `codex/level2_lea_rna_counterfactual`.
- Provenance HEAD: `4424e4acbbcc593f61b7556282f45f12e29802fc`.
- Fixed result commit: `c3d074412ecc1be0958141726db16a1ed0036a04`.
- Fixed OOF source: `results/level2_lea/LEVEL2_OOF_FROZEN.npz`.
- Fixed OOF SHA-256: `f65e373534dbb1b1a89b50a69907251a4ea608dd2d5d432cf704aa6795b27284`.
- Baseline shape: `(342, 10157)`.
- Biological-LCL outer-fold counts: `(np.int64(69), np.int64(69), np.int64(68), np.int64(68), np.int64(68))`.
- Real replicate reliability: `0.744254288540637`; 95% CI `[0.683605409015928, 0.7833655620937736]`.
- Six-model numerical tolerance: `1e-12`.

## Fixed real-result reconciliation

| model | observed real OOF R2 | fixed expected | absolute difference | pass |
|:--|--:|--:|--:|:--:|
| ridge | 0.002898149495299 | 0.002898149495299 | 9.584e-17 | True |
| pca_ridge | 0.002390466879790 | 0.002390466879790 | 5.421e-17 | True |
| pilot_mlp | 0.000161495537051 | 0.000161495537051 | 7.034e-17 | True |
| rbf_kernel_ridge | 0.003543750822482 | 0.003543750822482 | 1.691e-17 | True |
| hist_gradient_boosting | -0.003216159070450 | -0.003216159070450 | 8.674e-18 | True |
| deep_residual_mlp | -0.006316876159124 | -0.006316876159124 | 2.949e-17 | True |

All six values, the exact 342 by 10,157 baseline matrix, and the five fixed LCL-disjoint folds passed. No synthetic outcome was generated or inspected during this gate.
