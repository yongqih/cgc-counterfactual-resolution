# STATE independent Tahoe held-context artifact stability

Capability: `OFFICIAL_ARTIFACT_SUMMARY`. Source: `490f3af2d199c51a6fd3031ef8f087076d3ba9ab`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py state_independent_stability --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py state_independent_stability --entrypoint scripts/cgc_state_stability_audit.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/state_stability_official` — Official Arc STATE Tahoe evaluation artifacts with source URLs and hashes in STATE_STABILITY_ARTIFACT_MANIFEST.json. Size: not available bytes. not bundled; third-party terms or separate frozen-input archive required.
- `results/cgc_state_1/context_heterogeneity.csv` — Frozen discovery report overlay only. Size: 1795 bytes. not bundled; third-party terms or separate frozen-input archive required.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Use --analyze-only to avoid network/download path. No STATE training.
- Final source contains arithmetic held-target-truth exclusion fix 2c26f98a7dd839ee14cf4d52e0e2da07151c4580; original implementation e4d9f977 is not final authority.
- Five held contexts, 1136 interventions, 2000 features; official artifact-summary availability does not imply training deployment readiness.
