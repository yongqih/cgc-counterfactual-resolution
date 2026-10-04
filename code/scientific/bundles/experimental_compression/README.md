# Strict target-hidden entrywise Experimental Compression

## Execution

```text
python code/scientific/launch.py experimental_compression --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/cgc_entrywise_foundation.py`
- `scripts/cgc_entrywise_surface.py`
- `scripts/cgc_entrywise_controls.py`
- `scripts/cgc_entrywise_inference.py`
- `scripts/cgc_entrywise_report.py`

## Inputs

- `data/cgc_entrywise_cache/same_plate6_float32.npy`: ENTRYWISE_FOUNDATION_MANIFEST.json
- `data/cgc_entrywise_cache/same_plate14_float32.npy`: ENTRYWISE_FOUNDATION_MANIFEST.json
- `data/cgc_entrywise_cache/cross_plate6_plate14_float32.npy`: ENTRYWISE_FOUNDATION_MANIFEST.json
- `data/cgc_entrywise_cache/sample_gene_sums_float64.npy`: ENTRYWISE_FOUNDATION_MANIFEST.json

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Hyperparameter selection uses only the reference contexts observed at each support budget.
- Prediction reconstruction preserves support-sequence-specific parameters.
