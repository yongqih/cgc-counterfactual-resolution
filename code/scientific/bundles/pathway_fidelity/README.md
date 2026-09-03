# Fixed PROGENy pathway-specific fidelity and recoverability diagnostic

Capability: `FIGURE_FROM_FROZEN_PREDICTIONS`. Source: `ab44618011cd963cadfcc0702d9c9ec71b87a8c3`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py pathway_fidelity --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py pathway_fidelity --entrypoint scripts/progeny_pathway_fidelity_audit.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/cgc_resolution2_replay/truth_g_primary_float32.npy` — Frozen Tahoe replicate truth. Size: 955854128 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_resolution2_replay/m49_k92_predictions_float32.npy` — Frozen all-but-one model predictions; no refit. Size: 955854128 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_resolution2_replay/m49_k92_frozen_weights.npz` — Frozen prediction replay episode weights. Size: 3455810 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_resolution2_replay/hierarchical_bootstrap_weights_int16.npy` — Frozen shared hierarchical bootstrap weights. Size: 93000128 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_entrywise_compression/_cache/frozen_utility_table.npz` — Frozen entrywise utility and support/sentinel assignment. Size: 27158852 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_bio1_official/frozen_program_weights.parquet` — Frozen PROGENy weights and gene identifiers. Size: 2705687 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/cgc_bio1_official/progeny` — Official PROGENy source revision cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f version 1.17.3. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_tahoe_0i/gene_metadata_frozen.csv` — Frozen gene metadata axis. Size: 5092546 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_tahoe_0i/gene_indices_g_primary.npy` — Frozen primary gene indices. Size: 102908 bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Recovery is undefined where replicate cross-product denominator is nonpositive (e.g. VEGF); this is not a missing plotting point.
- Default --source-root is an author-machine Windows path; supply the arranged frozen-input root explicitly.
