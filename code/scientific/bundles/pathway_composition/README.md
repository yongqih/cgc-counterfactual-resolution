# All 16383 subsets of 14 predefined PROGENy pathways: fidelity and recoverability

Capability: `FIGURE_FROM_FROZEN_PREDICTIONS`. Source: `ce1a35649d80205035231e5fc305e857c79acfa1`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The source commit above identifies the historical base. On 2026-09-08, episode-reference-only hyperparameter selection and its downstream replay were corrected. SOURCE_MANIFEST.json distinguishes corrected bytes from the original Git blobs and retains their pre-correction hashes. Historical result tables must not be substituted for corrected results. The public correction and reproduction guide is ../../SUPPORT_BUDGET_CORRECTION.md.

## Safe inspection

```text
python code/scientific/launch.py pathway_composition --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. The 2026-09-08 corrected runs and unchanged analyses are distinguished in ../../SUPPORT_BUDGET_CORRECTION.md.

## Entry points

- `python code/scientific/launch.py pathway_composition --entrypoint scripts/pathway_resolution_scaling_audit.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/cgc_resolution2_replay/truth_g_primary_float32.npy` â€” Frozen Tahoe replicate truth. Size: 955854128 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_resolution2_replay/m49_k92_predictions_float32.npy` â€” Frozen all-but-one model predictions; no refit. Size: 955854128 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_resolution2_replay/m49_k92_frozen_weights.npz` â€” Frozen prediction replay episode weights. Size: 3455810 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_resolution2_replay/hierarchical_bootstrap_weights_int16.npy` â€” Frozen shared hierarchical bootstrap weights. Size: 93000128 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_entrywise_compression/_cache/frozen_utility_table.npz` â€” Frozen entrywise utility and support/sentinel assignment. Size: 27158852 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_bio1_official/frozen_program_weights.parquet` â€” Frozen PROGENy weights and gene identifiers. Size: 2705687 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_bio1_official/progeny` â€” Official PROGENy source revision cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f version 1.17.3. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_tahoe_0i/gene_metadata_frozen.csv` â€” Frozen gene metadata axis. Size: 5092546 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_tahoe_0i/gene_indices_g_primary.npy` â€” Frozen primary gene indices. Size: 102908 bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- No model fit; computes full subset projection and 10000 frozen-prediction bootstrap statistics, not mere CSV checks.
- Claim is composition-dependent recoverability, not a universal composition ceiling. Nonpositive denominators remain undefined.
- Default --source-root is an author-machine Windows path; explicit input root is required for portability.
