# Lea RNA-only strict LCL-disjoint pilot and six-model battery (ARCHS4 dynamic source authority)

Capability: `FULL_ANALYSIS`. Source: `c3d074412ecc1be0958141726db16a1ed0036a04`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py lea_historical_analysis --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py lea_historical_analysis --entrypoint scripts/level2_lea_pilot.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py lea_historical_analysis --entrypoint scripts/level2_lea_full_battery.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/level2_lea_official/GSE207049_31Mar21_all_runs_voom_resid.txt.gz` — GEO GSE207049 official author residualized processed matrix. Size: 217469362 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/level2_lea_official/31Mar21_all_runs_voom_resid_metadata.txt` — Exact sample-to-LCL/ETOH/DEX mapping in integrity and pairing audit. Size: 292921 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/level2_lea/LEVEL2_OOF_FROZEN.npz` — Pilot output required by full battery; six-model battery reuses pilot predictions/checkpoints. Size: 90490423 bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Raw reads not required. This representation is author post-SVA, not original HTSeq counts.
- Five strict LCL-disjoint folds; full battery requires an already completed pilot, not a new random split.
- Torch optional dependency required for neural models; no data or checkpoint arrays in source bundle.
