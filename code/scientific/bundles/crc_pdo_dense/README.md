# Dense conditioning-resolution saturation with matched-random controls

Capability: `FULL_ANALYSIS`. Source: `76478cd5f30851c7376b7aa95a171fcfb7472645`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py crc_pdo_dense --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py crc_pdo_dense --entrypoint scripts/run_crc_pdo_dense_resolution_plateau.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz` — GSE294511 processed RNAseq and official PDO/patient mapping; source_manifest.json. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz` — Kryeziu et al. Cell Reports Medicine 2026, DOI 10.1016/j.xcrm.2026.102840; official supplementary DSS and Mendeley 10.17632/hr94h42xdc.3. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/crc_pdo_personalized_drug_application` — Frozen premodel patient/panel/intersection audit tables required by application and follow-up runners. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/crc_pdo_pca_resolution_sweep` — Prior frozen OOF: FIXED_K_OOF_PREDICTIONS.csv, PCA_STAR_OOF_PREDICTIONS.csv, FULL_RNA_REOPT_OOF.csv and FINAL_QA; result 1bb0e95b39c2181638d732097f8f868689f32c87, ledger 624e6f239459418f8d6a8792ff41d2dcc1c2c61c. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Dense fixed grid2,4,8,12,14,16,18,20,22,24,26,28,30,32; nested patient-disjoint selection; 100 matched-random controls at16,24,32 and10000 bootstrap draws.
- Runner validates prior result/ledger Git references and hashes. A source-only export is not automatically a compatible full historical worktree.
- No new model fit was executed for this release packaging.
