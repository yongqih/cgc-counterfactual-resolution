# Additive versus CP low-rank interaction completion

## Execution

```text
python code/scientific/launch.py low_rank_completion --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/cgc_lowrank_run.py`
- `scripts/cgc_lowrank_masks.py`
- `scripts/cgc_lowrank_report.py`

## Inputs

- `data/cgc_entrywise_cache`: LOW_RANK_INTERACTION_PROTOCOL.md; ENTRYWISE_FOUNDATION_MANIFEST.json

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Fits additive and CP-interaction models using the recorded target-hidden masks.
