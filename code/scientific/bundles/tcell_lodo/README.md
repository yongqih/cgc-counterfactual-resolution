# Strict leave-one-donor-out T-cell predictor benchmark

## Execution

```text
python code/scientific/launch.py tcell_lodo --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `src/igc_virtual_cell/cgc_tcell_1b/benchmark.py`

## Inputs

- `data/processed/cgc_tcell`: T-cell truth freeze 4f2e608d9814015de98cfbc9cbf358af54fbc45e
- `results/cgc_tcell`: Frozen truth manifests, strict-trans genes, dataset_inventory.json and NTC manifest
- `data/raw/cgc_tcell/GWCD4i.pseudobulk_merged.h5ad`: Required by NTC baseline extraction if data/processed/cgc_tcell_1b/ntc_context.float32.npy is not already valid

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Requires CUDA-enabled PyTorch, a compatible GPU and prepared input caches.
- Uses four donor-disjoint outer folds.
- Response coefficients are computed by direct projection without subtracting the saved PCA mean; the run manifest records the projection convention.
