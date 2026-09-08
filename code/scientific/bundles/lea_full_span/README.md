# Fold-local full-training-span PCA oracle transfer sanity analysis

Capability: `FULL_ANALYSIS`. Source: `f207a2278a76e51ada5fcb600d7049ac03d45087`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py lea_full_span --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py lea_full_span --entrypoint scripts/run_full_span_pca_sanity_audit.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/wgcna_frontier_official/MATCHED_POSTSVA_340_REPRESENTATION.npz` — Matched Lea 340-LCL response representation and frozen folds. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/wgcna_frontier_official/MATCHED_POSTSVA_FULL_OOF.npz` — Matched post-SVA six-model OOF archive. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/level2_lea/archs4_robustness/ARCHS4_GENE_AXIS_RECONCILIATION.csv` — Frozen matched gene axis. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_dimension_matched_compression` — Historical PCA curve/summary comparison required by reporting. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Fits fold-local PCA only; it is an oracle geometry diagnostic, not a new prediction model.
- run_wgcna_resolution_frontier.py is a dynamic data-loader authority required by the unchanged script; no WGCNA run or WGCNA manuscript claim is authorized.
- Historical directory names remain unchanged to preserve exact source. They do not reintroduce abandoned WGCNA experiments.
