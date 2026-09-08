# Strict leave-one-donor-out T-cell predictor benchmark

Capability: `FULL_ANALYSIS`. Source: `f8f6b345fba8b90b834558834f26665b63300177`.

Runtime: **ORIGINAL_HISTORY_AND_CUDA_REQUIRED; not runnable from code-only ZIP**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py tcell_lodo --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py tcell_lodo --entrypoint src/igc_virtual_cell/cgc_tcell_1b/benchmark.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/processed/cgc_tcell` — T-cell truth freeze 4f2e608d9814015de98cfbc9cbf358af54fbc45e. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_tcell` — Frozen truth manifests, strict-trans genes, dataset_inventory.json and NTC manifest. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/raw/cgc_tcell/GWCD4i.pseudobulk_merged.h5ad` — Required by NTC baseline extraction if data/processed/cgc_tcell_1b/ntc_context.float32.npy is not already valid. Size: 44566657140 bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- CUDA-enabled Torch and a compatible GPU are required by this unchanged implementation; benchmark.run explicitly rejects unavailable CUDA. Substantial input caches are required.
- Four donor-disjoint outer folds; response-basis protocol deviation is explicitly disclosed in original run_manifest.json, not hidden by source packaging.
- Original-history gate is retained, not bypassed or simulated. Execution needs the recorded original branch, ancestor commits and unmodified frozen-result paths; SOURCE_MANIFEST cannot substitute for those Git checks. CUDA is an explicit original implementation requirement. The original-history requirement is an external execution dependency, not a claim of standalone ZIP executability.
