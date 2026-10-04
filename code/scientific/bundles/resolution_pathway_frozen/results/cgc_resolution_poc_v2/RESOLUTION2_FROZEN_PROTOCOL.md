# Gene and pathway response comparison

The analysis evaluates the same context-specific predictions in 25,695 gene coordinates and five externally defined PROGENy pathway coordinates. Predictions, pathway membership and pathway weights are fixed before scoring.

## Data and estimator

- Estimator: `M2_AFFINE_RIDGE`.
- Primary support: `m=49,k=92`; secondary support: `m=40,k=4`.
- Target: context-specific response relative to the observed reference contexts.
- Plate 6 and Plate 14 are fitted separately and paired during evaluation.
- Gene axis: the ordered `G_PRIMARY` axis of 25,695 genes.
- Source identifiers and input checksums are recorded in `SOURCE_MANIFEST.json`.

## Prediction reconstruction

Reconstruction loads the recorded plate-, target-, support- and budget-specific parameters. It uses the corresponding split manifest, support and sentinel orders, preprocessing, responses, Gram arrays and seeds. Hyperparameters are selected using only the observed reference set, as specified in `code/scientific/SUPPORT_BUDGET_CORRECTION.md`.

For reference contexts `S_m` and fitted weights `w`, the excess prediction is

`Dhat[p,c] = sum_s (w[p,s] - 1/m) * Delta[p,s]`.

The paired response is

`D[p,c] = Delta[p,c] - mean_s Delta[p,s]`.

At `(40,4)`, all eight support sequences are reconstructed separately. Paired utility contributions are averaged within each target entry before bootstrap resampling. At `(49,92)`, the support sequences share one all-but-one information set.

## Integrity checks

Reconstructed predictions must reproduce the recorded recovery summaries, preserve all context/intervention/plate/gene axes, and have finite paired denominators. Repeated reconstruction must give identical vectors. Replacing hidden target outcomes must leave support identities, hyperparameters, weights and predictions unchanged.

## Pathway coordinates

The panel comprises `MAPK`, `PI3K`, `JAK-STAT`, `p53` and `NFkB`. All 93 interventions are evaluated on all five axes. The signed PROGENy vectors are mapped to `G_PRIMARY`, with zero weight for absent genes and unit L2 normalization. The same vector projects each observed and predicted response. No pathway-specific predictor is fitted.

Recovery pools the five coordinates using the paired replicate-stable ratio of summed energies. The primary comparison is pooled pathway-minus-gene recovery.

## Inference

Use the recorded 10,000 hierarchical paired bootstrap draws. Resample contexts and then interventions within sampled contexts, preserving plate pairing. Support-sequence contributions are averaged before resampling. Confidence intervals use centered studentized maximum-absolute-deviation calibration across the specified four-term family. The two pathway-minus-gene contrasts are the biological comparisons; random-panel terms contribute to calibration.

Pathway-coordinate recovery and orthogonal-span fidelity have different quadratic forms. Their product is not interpreted as a fraction of full-gene signal recovered. The source data report these quantities separately.
