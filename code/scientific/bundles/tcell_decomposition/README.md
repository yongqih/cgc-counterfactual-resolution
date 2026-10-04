# T-cell response alignment and shared versus donor-specific operators

## Execution

```text
python code/scientific/launch.py tcell_decomposition --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `src/igc_virtual_cell/cgc_tcell_1d/decompose.py`

## Inputs

- `results/cgc_tcell_1b`: Frozen 1B predictions and model checkpoint manifest at f8f6b345fba8b90b834558834f26665b63300177
- `data/processed/cgc_tcell`: Truth freeze 4f2e608d9814015de98cfbc9cbf358af54fbc45e

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Computes response alignment and shared versus donor-specific components from saved predictions.
- Execution requires recorded Git source objects and input hashes.
- Intervention bootstrap draws are conditional on the four donors.
