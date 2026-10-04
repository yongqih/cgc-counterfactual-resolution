# Full-training-span PCA sanity protocol

Fixed before computing any held full-span outcome.

## Provenance and scope

- Fixed dimension-matched analysis parent: `77b346e669b5daf4a4492816cee3429337b4d9ff`.
- Matched-340 representation SHA-256: `7f44947ec59a43eb954ec2032e4b9c3733306542bcd117e03f584d72b3ecf59b`.
- Fixed global-PCA curve SHA-256: `31c7e72d76becb9fd78f2f52ad85fd8dff5f91195981f70649bfd78bac4d21e6`.
- Fixed dimension-matched summary SHA-256: `220b98089ff92008b86b94920127ec322a547ae27995fe414e1920960c565516`.
- Data, sample order, recorded five biological-LCL-disjoint folds, 10,110 genes, ETOH–DEX response, training-shared centering and held full-gene denominator define the analysis.

## Exact SVD matrix and numerical rank

For each outer fold, start with the exact float32 `y_train` supplied to the fixed global-PCA analysis. As standard PCA does, calculate its outer-training column mean in float64 and form:

`X_train = y_train.astype(float64) - mean_train`.

Compute the direct thin SVD `X_train = U S V^T` with `numpy.linalg.svd(full_matrices=False)`. Numerical rank is the number of singular values exceeding:

`max(n_train, 10110) × eps(float64) × S_max`.

This is the same prespecified rank tolerance used in the fixed dimension-matched analysis. Do not force rank to `n_train−1`.

## Reconstruction and recorded fidelity

The full-span affine PCA reconstruction is:

`prediction = mean_train + (response - mean_train) V_r^T V_r`.

Training full-rank reconstruction reports relative squared Frobenius error against the exact supplied `y_train`, maximum absolute error and the fixed fidelity:

`F = 1 - sum(||truth - prediction||²) / sum(||truth||²)`.

Require training `F >= 1−1e−12`, relative error `<=1e−12`, and maximum right-basis Gram error `<=1e−10` before held interpretation.

Direct SVD prefixes at `D=4,8,16,32,64,128,256` use the same affine reconstruction and must reproduce the fixed recorded D=256 fold and pooled values within absolute `1e−10`.

## Full-span held result and pure geometry

Use every non-zero training right singular vector in each fold. The primary recorded-metric quantity is the pooled full-span affine fidelity. The independent geometric quantity uses the zero-origin definition on the exact train-shared-centered held response `r_i = y_test[i]`:

`q_i = ||P r_i||² / ||r_i||²`, where `P = V_r^T V_r`.

For every held LCL, verify:

`||r_i||² = ||P r_i||² + ||(I−P)r_i||²`

within relative `1e−10`. Report raw `q_i`, and additionally report the affine-centered fraction using `r_i−mean_train` so the recorded fidelity can be reconciled exactly.

The fixed fidelity and raw pooled projection fraction use different centering conventions. Algebraically:

`F_affine = 1 - E_perp,affine / E_raw`,

whereas:

`q_raw = E_parallel,raw / E_raw`.

Their difference is caused only by the training-mean origin and its cross terms. Also report:

`q_affine = E_parallel,affine / E_affine`.

## Training variance and optional LOO

Report cumulative training singular-value energy at `D=4,8,16,32,64,128,256` and full numerical rank. These are training variance fractions, not held fidelities.

Leave-one-training-LCL span estimation was not evaluated. The analysis uses the five outer training folds.

## Numerical validation

Use exactly one verdict:

- `HELDOUT_RESPONSE_SUBSPACE_TRANSFER_POOR` if training reconstruction, direct D256 reproduction and projection identities pass but full-span held transfer remains low;
- `GLOBAL_PCA_METRIC_OR_IMPLEMENTATION_ISSUE` if any algebraic or recorded-reproduction gate fails;
- `GLOBAL_PCA_SANITY_AUDIT_AMBIGUOUS` only if numerical or provenance limitations prevent adjudication.
