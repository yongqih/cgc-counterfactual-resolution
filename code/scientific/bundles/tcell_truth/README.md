# Primary CD4 T-cell pseudobulk truth construction and guide-disjoint reliability

Capability: `FULL_ANALYSIS`. Source: `4f2e608d9814015de98cfbc9cbf358af54fbc45e`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py tcell_truth --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py tcell_truth --entrypoint src/igc_virtual_cell/cgc_tcell/prepare.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py tcell_truth --entrypoint src/igc_virtual_cell/cgc_tcell/guide_disjoint.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py tcell_truth --entrypoint src/igc_virtual_cell/cgc_tcell/analysis.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/raw/cgc_tcell/GWCD4i.pseudobulk_merged.h5ad` — Official processed guide x donor x culture pseudobulk; exact URL and checksum in download_manifest.json. Size: 44566657140 bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- No per-cell prediction. Guide-disjoint donor-state contrasts are reconstructed from official pseudobulk input.
