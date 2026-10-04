# Lea semi-synthetic baseline-encoded response identifiability control

## Execution

```text
python code/scientific/launch.py lea_semisynthetic --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/lea_semisynthetic_control.py`

## Inputs

- `results/level2_lea/LEVEL2_OOF_FROZEN.npz`: Historical response/base-state arrays and fixed LCL folds
- `results/level2_lea/FULL_MODEL_BATTERY_RESULTS.csv`: Frozen real-data results for side-by-side comparison only

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- The simulated response has latent rank 4, fixed replicate reliability and ten simulation seeds. It tests pipeline sensitivity to baseline-encoded signals.
- Hyperparameters come from the real-data analysis. The tiered model and signal-strength comparison runs on CPU or CUDA.
