# Gene and pathway readouts of the same response predictions

## Execution

```text
python code/scientific/launch.py resolution_pathway_frozen --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Analysis entrypoint: `render_valid_resolution.py`.

## Inputs

- `results/cgc_resolution_poc_v2/RESOLUTION2_PAIRED_COMPARISONS.csv`: 46b25eec4a97e3001cc43262980af8f69c533478; valid contrast selected under21b08f02d3371129bb5cbf49704c65a271240c7c

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- The pathway-minus-gene comparison uses fixed predictions and a four-term max-T calibration.
- The primary recovery estimates are 0.27281393788398667 in pathway coordinates and 0.11068053088808294 in gene coordinates.
- Use the release entrypoint listed in this manifest and supply source and truth roots explicitly.

Execution rule: The launcher executes the release_entrypoint listed in this manifest.
