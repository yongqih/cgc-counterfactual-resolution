# Predefined five-pathway held-context scaling

## Execution

```text
python code/scientific/launch.py five_pathway_scaling --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/tahoe_five_pathway_scaling.py`

## Inputs

- `results/cgc_tahoe_0i/response_tensors.zarr`: response_manifest.json
- `data/cgc_bio1_official/frozen_program_weights.parquet`: fixed pathway model source

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Five pathways are defined independently of outcomes.
- Compute hierarchical intervals using ed7_hierarchical_repair.

Execution rule: The launcher permits the prepare stage. Use ed7_hierarchical_repair for hierarchical inference.
