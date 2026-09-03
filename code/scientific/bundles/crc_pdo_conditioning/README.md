# Training-local biological conditioning-resolution comparison

Capability: `FULL_ANALYSIS`. Source: `f3722de40a4b65d683d758ad9e6634a0443cc759`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py crc_pdo_conditioning --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py crc_pdo_conditioning --entrypoint scripts/run_crc_pdo_conditioning_resolution.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz` — GSE294511 processed RNAseq and official PDO/patient mapping; source_manifest.json. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz` — Kryeziu et al. Cell Reports Medicine 2026, DOI 10.1016/j.xcrm.2026.102840; official supplementary DSS and Mendeley 10.17632/hr94h42xdc.3. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/crc_pdo_personalized_drug_application` — Frozen premodel patient/panel/intersection audit tables required by application and follow-up runners. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `PROGENy weights file` — Official PROGENy v1.17.3 revision cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- All transforms and model choices are training-local; no target DSS enters features.
- HTA biological projection is NOT_EXECUTABLE without official probe-to-gene annotation; do not infer that annotation or claim successful HTA pathway validation.
