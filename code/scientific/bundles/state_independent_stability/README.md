# STATE independent Tahoe held-context artifact stability

## Execution

```text
python code/scientific/launch.py state_independent_stability --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/cgc_state_stability_audit.py`

## Inputs

- `data/state_stability_official`: Official Arc STATE Tahoe evaluation artifacts with source URLs and hashes in STATE_STABILITY_ARTIFACT_MANIFEST.json
- `results/cgc_state_1/context_heterogeneity.csv`: Frozen discovery report overlay only

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Use --analyze-only when input artifacts are already available.
- The held target is excluded before arithmetic operations that construct the comparison mean.
- Five contexts, 1,136 interventions and 2,000 features are evaluated using official prediction artifacts.
