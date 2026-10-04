# Target-hidden Tahoe context-support scaling

## Execution

```text
python code/scientific/launch.py held_context_scaling --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/tahoe_held_context_scaling.py`
- `scripts/tahoe_held_context_secondary.py`

## Inputs

- `results/cgc_tahoe_0i/response_tensors.zarr`: response_manifest.json
- `data/tahoe100m_control_state/control_wells_log1p_cpm_float32.npy`: fixed control-state source manifest
- `results/cgc_tahoe_0d/lineage_baseline.csv`: fixed lineage metadata

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- The target context is excluded from fitting.
- Compute context-then-intervention bootstrap intervals using ed7_hierarchical_repair.

Execution rule: Use ed7_hierarchical_repair for hierarchical inference.
