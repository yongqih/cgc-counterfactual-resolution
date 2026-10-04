# Observed-support hyperparameter selection

Each Experimental Compression support path selects hyperparameters using only its response-observed reference contexts. Tuning pools support paths only when their observed reference sets are identical. Downstream prediction reconstruction retains the support-sequence index.

The implementation uses protocol identifier `EPISODE_REFERENCE_SUPPORT_ONLY_V2`. Cache metadata must match the implementation, support set and parameter selection recorded in the source manifests.

## Execution

Provide the input paths listed in `INPUT_CONTRACTS.json`, including the foundation and split manifests. Use a separate output workspace.

```text
python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_surface.py --execute -- run --root ROOT --target-start 0 --target-stop 50
python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_surface.py --execute -- aggregate --root ROOT
python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_inference.py --execute -- --root ROOT
```

The result directory contains the recovery surface, thresholds, confidence intervals, null summaries and checksums. Per-target caches, expression arrays and prediction tensors are additional inputs.

## Pathway comparisons

`ReplayContext(ROOT, ROOT)` and `materialize_predictions` in `cgc_resolution_poc.replay` reconstruct gene-level predictions from the recorded parameters. Supply the response tensors, split metadata, per-target caches and Gram arrays. The `budgets` argument selects which support settings to reconstruct.

`recompute_corrected_pathway.py --help` describes the paired-inference entrypoint. It requires prediction vectors, pathway weights, bootstrap weights, calibration membership and candidate-score arrays. It computes pathway-minus-gene comparisons at two support settings, with the four-term max-T simultaneous calibration specified in the analysis.

The figure renderer reads `RESOLUTION2_VALID_COMPARISONS_CORRECTED.csv`. Random-panel terms contribute to interval calibration; biological interpretation uses the pathway-minus-gene comparisons.
