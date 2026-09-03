# STATE official held-context artifact geometry summary

Capability: `OFFICIAL_ARTIFACT_SUMMARY`. Source: `58e7facf1b9b9573ee487469afdecfdff6f1eece`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py state_official_summary --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py state_official_summary --execute -- --official-data-dir <official-data-dir> --out <new-summary.json>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/state_official_predictions` — ArcInstitute/state; HuggingFace ST-SE-Parse and ST-HVG-Parse official eval-best full-DE artifacts; 434168727 bytes recorded in source manifest. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Official artifact summary, not STATE model training or recreation of the official upstream predictions.
- DO NOT run --help: this historical script has no argparse guard and would execute analysis.
- Historical sampled context-null p-value is superseded in manuscript by conservative exact-unique-support 1/45=0.022222; unmodified legacy source may reproduce obsolete 1/5001. Frozen audit authority 21b08f02d3371129bb5cbf49704c65a271240c7c governs interpretation.
- claim_boundary: Current summary reads actual official mean artifacts, computes existing audited metrics, and uses all44 exact context derangements with conservative plus-one p. Historical sampled p and historical main/rendering are not active; no State fitting or bootstrap intervals are claimed by the bounded summary.
- active_execution_guard: Original all-stages scripts cannot be invoked through release launcher; only current bounded entrypoint is allowed.
