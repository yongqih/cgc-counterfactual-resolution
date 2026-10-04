# Dense conditioning-resolution saturation with matched-random controls

## Execution

```text
python code/scientific/launch.py crc_pdo_dense --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/run_crc_pdo_dense_resolution_plateau.py`

## Inputs

- `data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz`: GSE294511 processed RNAseq and official PDO/patient mapping; source_manifest.json
- `data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz`: Kryeziu et al. Cell Reports Medicine 2026, DOI 10.1016/j.xcrm.2026.102840; official supplementary DSS and Mendeley 10.17632/hr94h42xdc.3
- `results/crc_pdo_personalized_drug_application`: Patient patient/panel/intersection analysis tables required by application and follow-up runners
- `results/crc_pdo_pca_resolution_sweep`: Prior fixed OOF: FIXED_K_OOF_PREDICTIONS.csv, PCA_STAR_OOF_PREDICTIONS.csv, FULL_RNA_REOPT_OOF.csv and FINAL_QA; result 1bb0e95b39c2181638d732097f8f868689f32c87, ledger 624e6f239459418f8d6a8792ff41d2dcc1c2c61c

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Component counts: 2, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 and 32. Selection is nested within patient-disjoint folds.
- There are 100 dimension-matched random controls at 16, 24 and 32 components and 10,000 bootstrap draws.
- Execution requires the recorded input predictions, Git source objects and matching hashes.
