# Tahoe replicate-core inventory, pseudobulk extraction and truth

## Execution

```text
python code/scientific/launch.py tahoe_replicate_core --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/cgc_tahoe_0b_inventory.py`

## Inputs

- `data/tahoe100m_plate6_14_core`: results/cgc_tahoe_0b and cgc_tahoe_0c fixed manifests

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Constructs paired-plate pseudobulk responses and their reproducibility summaries.
