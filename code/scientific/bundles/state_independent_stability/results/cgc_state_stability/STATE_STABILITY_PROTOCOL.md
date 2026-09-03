# STATE stability protocol (frozen before outcome calculation)

Authority commit: `35cf07d8f3d90450b365893b56aa127dca7c4774`  
Discovery evaluator: `src/igc_virtual_cell/cgc_state_1/analysis.py` and `scripts/cgc_state_1_analysis.py` at the authority commit  
Selected independent setting: official Tahoe-100M zero-shot  
Primary model artifact: ST-SE-Tahoe `eval_best.ckpt`, revision `03b1971d7cc93a7535fd2e957c6948dba267378b`  
Prespecified sensitivity: ST-HVG-Tahoe `eval_best.ckpt`, revision `ca6b751972493f8448e3256d1340ae70ad43e1e7`

## Frozen matrix

- Contexts: C32, HOP62, HepG2-C3A, Hs 766T, PANC-1.
- Interventions: the sorted exact string intersection across all five official real and predicted DE tables, expected `n=1,136` drug-dose-unit tuples. No outcome-based filtering.
- Features: exact official integer feature axis 0–1999. Every context must contain all 2,000 features for every selected intervention.
- Truth response: official `target_mean - reference_mean` from the context's `*_real_de.csv`.
- STATE response: official predicted `target_mean` minus the context-matched real DMSO reference mean. This is the same control convention used in CGC-STATE-1.
- Shared-rule comparator: leave-one-context-out mean of true responses for the same intervention. The target context is masked before any summation or averaging (rather than computed as `sum(all)-target`) so even floating-point roundoff cannot carry target truth into its comparator. This is mathematically the same frozen LOCO object. No STATE prediction is used in this comparator.

## Frozen evaluator

Reuse the exact functions in `src/igc_virtual_cell/cgc_state_1/analysis.py` for leave-one-context means, recovery statistics, hierarchical context/intervention bootstrap, correspondence nulls, and context geometry. The primary statistic remains

`g = 1 - SSE_STATE / SSE_PertMean`

with the same aggregate and per-context construction as CGC-STATE-1. The audit also reports the existing alpha, kappa, cosine, shared/novel decomposition, delta-Pearson, and delta-MSE fields. No CellEval metric can substitute for `g`.

## Frozen inference and verdict interpretation

- Primary stability units are five held contexts. Report each `g_c`, positive fraction, median, range, negative contexts, and the exact existing hierarchical context/intervention bootstrap for the aggregate.
- ST-SE determines the verdict. ST-HVG is labeled sensitivity and cannot rescue or reverse the primary verdict.
- `STATE_OPERATOR_RECOVERY_BROADLY_STABLE`: at least four of five contexts have `g_c>0`, the context median is at least 0.10, and no context has `g_c<-0.10`.
- `STATE_OPERATOR_RECOVERY_HETEROGENEOUSLY_REPLICATED`: at least two contexts have `g_c>0.10` and at least one context has `g_c<-0.10`. This deliberately requires both substantial recovery and material negative transfer rather than classifying numerical sign noise as heterogeneity.
- `STATE_OPERATOR_RECOVERY_NOT_REPLICATED`: neither of the preceding rules is met; this includes predominantly null/negative outcomes and uniformly weak positive values.
- `STATE_STABILITY_AUDIT_NOT_EXECUTABLE`: artifact, correspondence, feature-axis, or leakage gate fails before valid inference.

These thresholds are frozen before Tahoe DE outcomes are loaded. The aggregate hierarchical-bootstrap interval is reported as uncertainty, but with only five biological stability units it does not override the frozen context-pattern verdict rule.

## Leakage and integrity gates

1. Verify repository revision, LFS SHA-256, byte count, CSV schema, exact context names, exact intervention intersection, feature axis, and control label.
2. Verify official zero-shot run configuration and prohibit few-shot/eval-last/checkpoint mixing.
3. Recompute predictions only by reading immutable official predicted means; perform no fitting, calibration, centering, PCA, feature selection, or normalization using held-context truth.
4. Run a synthetic truth-mutation test: modifying a held-context truth tensor after artifact loading must leave the loaded prediction tensor, feature/intervention axes, and prediction hash unchanged.
5. Fail closed on any prediction/truth axis mismatch or non-finite response vector.

## Download and compute limits

Download only compact per-context `*_pred_de.csv`, `*_real_de.csv`, run configuration, and small metric tables. Do not download H5AD, checkpoints, raw cells, `eval_last`, or few-shot artifacts. Evaluation is CPU/vectorized and requires no model training or GPU.

## Stop rule

After one formal ST-SE outcome, the frozen ST-HVG sensitivity, reconciliation, one Extended Data figure, manuscript wording, tests, and commit, stop. Do not select another dataset, tune STATE, or investigate mechanisms.
