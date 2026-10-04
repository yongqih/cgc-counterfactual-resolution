# LCL-disjoint RNA-to-response prediction with six models

## Execution

```text
python code/scientific/launch.py lea_historical_analysis --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/level2_lea_pilot.py`
- `scripts/level2_lea_full_battery.py`

## Inputs

- `data/level2_lea_official/GSE207049_31Mar21_all_runs_voom_resid.txt.gz`: GEO GSE207049 official author residualized processed matrix
- `data/level2_lea_official/31Mar21_all_runs_voom_resid_metadata.txt`: Exact sample-to-LCL/ETOH/DEX mapping in integrity and pairing audit
- `results/level2_lea/LEVEL2_OOF_FROZEN.npz`: Pilot output required by full battery; six-model battery reuses pilot predictions/checkpoints

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Uses the author-provided post-SVA expression representation.
- Five biological-LCL-disjoint folds are shared by the pilot and full model comparison.
- Neural models require PyTorch.
