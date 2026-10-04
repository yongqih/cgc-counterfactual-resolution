# Training-local biological conditioning-resolution comparison

## Execution

```text
python code/scientific/launch.py crc_pdo_conditioning --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/run_crc_pdo_conditioning_resolution.py`

## Inputs

- `data/crc_pdo_personalized_application/processed/RNAseq_PDO_log2CPM1.npz`: GSE294511 processed RNAseq and official PDO/patient mapping; source_manifest.json
- `data/crc_pdo_personalized_application/processed/PRIMARY_DSS.npz`: Kryeziu et al. Cell Reports Medicine 2026, DOI 10.1016/j.xcrm.2026.102840; official supplementary DSS and Mendeley 10.17632/hr94h42xdc.3
- `results/crc_pdo_personalized_drug_application`: Patient patient/panel/intersection analysis tables required by application and follow-up runners
- `PROGENy weights file`: Official PROGENy v1.17.3 revision cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f; expected SHA256 59227c888c2bfa1ed89fe00109da7ce1bacffbb9b8b6b79215065d41f2476223

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Transforms and model choices use training data only; target DSS values are excluded from features.
- HTA pathway projection requires official probe-to-gene annotation.
