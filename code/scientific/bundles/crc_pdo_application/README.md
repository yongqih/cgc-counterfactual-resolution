# Patient-disjoint ex-vivo functional drug prioritization and nested Ridge

## Execution

```text
python code/scientific/launch.py crc_pdo_application --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/run_crc_pdo_personalized_audit.py`
- `scripts/run_crc_pdo_personalized_application.py`

## Inputs

- `data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz`: GSE294511 processed RNAseq and official PDO/patient mapping; source_manifest.json
- `data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz`: Kryeziu et al. Cell Reports Medicine 2026, DOI 10.1016/j.xcrm.2026.102840; official supplementary DSS and Mendeley 10.17632/hr94h42xdc.3
- `results/crc_pdo_personalized_drug_application`: Patient patient/panel/intersection analysis tables required by application and follow-up runners
- `data/crc_pdo_personalized_application/raw`: Official supplementary spreadsheets, GEO processed matrices and exact source-manifest hashes needed only for preparation
- `data/crc_pdo_personalized_application/processed/HTA2_PDO_public_processed.npz`: Secondary HTA expression platform; full original application loops over RNAseq and HTA2.0

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Primary RNA-seq cohort: 52 patients, 91 organoids, 24 drugs and 19,421 genes. The HTA expression platform is analyzed separately.
- Patient-level aggregation and leave-one-patient-out evaluation; alpha is selected within five training folds.
- Shared sign-flip draws use restudentization for simultaneous inference.
- The --postprocess-only option requires saved predictions and null draws.
