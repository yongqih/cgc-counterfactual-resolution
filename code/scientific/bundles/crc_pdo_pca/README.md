# Nested PCA conditioning-resolution sweep

## Execution

```text
python code/scientific/launch.py crc_pdo_pca --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/run_crc_pdo_pca_resolution_sweep.py`

## Inputs

- `data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz`: GSE294511 processed RNAseq and official PDO/patient mapping; source_manifest.json
- `data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz`: Kryeziu et al. Cell Reports Medicine 2026, DOI 10.1016/j.xcrm.2026.102840; official supplementary DSS and Mendeley 10.17632/hr94h42xdc.3
- `results/crc_pdo_personalized_drug_application`: Patient patient/panel/intersection analysis tables required by application and follow-up runners
- `results/crc_pdo_conditioning_resolution/PCA14_OOF.csv`: Frozen conditioning result 38adafd8601f3a0382baa88e03bf72e3dd0d3a62; accompanying CRC_PDO_CONDITIONING_RESOLUTION_FINAL_QA.json required

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- PCA and Ridge use leave-one-patient-out evaluation with five inner folds. Inner model selection minimizes training-validation MSE.
- The dense component sweep uses the saved out-of-fold predictions from this analysis.
