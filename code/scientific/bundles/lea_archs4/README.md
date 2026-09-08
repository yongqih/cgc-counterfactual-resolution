# Independent ARCHS4 pre-SVA quantification robustness

Capability: `FULL_ANALYSIS`. Source: `2fe0deaf3d7b07088488d3bcbe2676c46f2ab353`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py lea_archs4 --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py lea_archs4 --entrypoint scripts/run_lea_archs4_robustness.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py lea_archs4 --entrypoint scripts/prepare_lea_archs4_robustness.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/level2_lea_archs4_matched` — ARCHS4 v2.5 matched count extraction and prepared matched 340-LCL/10110-gene representations; exact hashes in extraction/run manifests. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/level2_lea/LEVEL2_OOF_FROZEN.npz` — Historical Lea frozen 342-LCL OOF and exact matched-core selection. Size: 90490423 bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/level2_lea_official/GSE207049_31Mar21_all_runs_voom_resid.txt.gz` — Matched post-SVA comparator from author GEO matrix. Size: 217469362 bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Dynamic git-show source loading requires historical c3d074 source objects. A separate lea_historical_analysis bundle supplies its unchanged local-import closure.
- Source-root is an explicit path; compiled historical modules may retain source-layout assumptions.
- ARCHS4 uniform Kallisto-derived counts are not Lea original HTSeq counts. No full ARCHS4 matrix/raw reads are downloaded by this packaging.
- Extraction entrypoints perform HTTP range access if executed; --help only is safe during packaging.
- path_adapter: Unchanged source requests historical modules through git_source; launcher maps only this artifact lookup to exact SHA-verified runtime_sources blobs. No incomplete Git database and no scientific function replacement.
