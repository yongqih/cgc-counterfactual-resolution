# Full-span PCA metric reconciliation

The frozen oracle fidelity is:

`F_affine = 1 - sum_i ||(I-P)(r_i-mu_train)||^2 / sum_i ||r_i||^2`.

The requested pure zero-origin geometric fraction is:

`q_raw,pooled = sum_i ||P r_i||^2 / sum_i ||r_i||^2`.

They are not algebraically identical because PCA uses the training-column mean `mu_train` as a fixed affine origin, whereas `q_raw` projects the already train-shared-centered response from zero. The corresponding centered geometric fraction is:

`q_affine,pooled = sum_i ||P(r_i-mu_train)||^2 / sum_i ||r_i-mu_train||^2`.

Exact results:

- frozen-metric full-span fidelity: `0.057844185760996`;
- pooled raw projection-energy fraction: `0.057844185788772`;
- pooled affine-centered projection-energy fraction: `0.057844185829461`;
- `F_affine - q_raw`: `-2.778e-11`;
- fidelity reconstructed independently from affine residual energy: `0.057844185760996`;
- maximum per-LCL raw projection-identity relative error: `4.187e-16`.

Thus any tiny fidelity-versus-`q_raw` difference is fully explained by the fixed training-mean origin, not by a metric or projector inconsistency.
