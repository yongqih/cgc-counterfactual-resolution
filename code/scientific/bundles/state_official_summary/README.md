# STATE official held-context artifact geometry summary

## Execution

```text
python code/scientific/launch.py state_official_summary --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Analysis entrypoint: `current_state_summary.py`.

## Inputs

- `data/state_official_predictions`: ArcInstitute/state; HuggingFace ST-SE-Parse and ST-HVG-Parse official eval-best full-DE artifacts; 434168727 bytes recorded in source manifest

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Evaluates official State predictions.
- Uses all 44 exact context derangements and a conservative plus-one probability with minimum 1/45.
- Use the release entrypoint listed in this manifest.

Execution rule: The launcher executes the release_entrypoint listed in this manifest.
