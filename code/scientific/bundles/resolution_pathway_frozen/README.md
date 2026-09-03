# Same frozen predictions, gene-versus-predefined-pathway readout and corrected inference

Capability: `FIGURE_FROM_FROZEN_PREDICTIONS`. Source: `46b25eec4a97e3001cc43262980af8f69c533478`.

Runtime: **FROZEN_CONTRAST_INPUT_INCLUDED; matplotlib/Arial runtime required; not rendered in this repair**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py resolution_pathway_frozen --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py resolution_pathway_frozen --execute -- --out-dir <new-output-dir>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `results/cgc_resolution_poc_v2/RESOLUTION2_PAIRED_COMPARISONS.csv` — 46b25eec4a97e3001cc43262980af8f69c533478; valid contrast selected under21b08f02d3371129bb5cbf49704c65a271240c7c. Size: not available bytes. included byte-exact frozen result table; active renderer discards every random-arm row.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Historical random-projection arm is INVALID and excluded from manuscript inference. Source inclusion is archival, not endorsement or permission to execute that arm.
- Valid pathway-minus-gene comparison is 0.27281393788398667 versus 0.11068053088808294; the claim authority is the corrected frozen interval, not random-projection results.
- Default source/truth roots are author-machine Windows paths; override them explicitly. No automatic replay or scientific execution during packaging.
- claim_boundary: Current analytical renderer selects only two frozen pathway-minus-gene contrasts and their unchanged corrected intervals. Invalid random-projection code/results cannot be executed or rendered by the active release entrypoint. No prediction or null is regenerated; manual R2 layout is not reproduced.
- active_execution_guard: Original all-stages scripts cannot be invoked through release launcher; only current bounded entrypoint is allowed.
