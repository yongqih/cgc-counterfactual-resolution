# Frozen T-cell predictions: alignment and shared-versus-donor-specific operators

Capability: `FIGURE_FROM_FROZEN_PREDICTIONS`. Source: `6605b095fc5d17863bec11233418f10819ca03a5`.

Runtime: **ORIGINAL_HISTORY_AND_CUDA_REQUIRED; not runnable from code-only ZIP**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py tcell_decomposition --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py tcell_decomposition --entrypoint src/igc_virtual_cell/cgc_tcell_1d/decompose.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `results/cgc_tcell_1b` — Frozen 1B predictions and model checkpoint manifest at f8f6b345fba8b90b834558834f26665b63300177. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/processed/cgc_tcell` — Truth freeze 4f2e608d9814015de98cfbc9cbf358af54fbc45e. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- No predictor fitting or inference; computes operator metrics and frozen-table/bootstrap adjudication.
- Integrity gate compares historical Git source paths. A source-only partial Git authority is not a full historical worktree; execution requires those original provenance refs and inputs.
- Final branch includes explicit bootstrap-unit correction and shared-operator serialization fixes; use final source, not initial 1D implementation.
- Original-history gate is retained, not bypassed or simulated. Execution needs the recorded original branch, ancestor commits and unmodified frozen-result paths; SOURCE_MANIFEST cannot substitute for those Git checks. CUDA is an explicit original implementation requirement. The original-history requirement is an external execution dependency, not a claim of standalone ZIP executability.
