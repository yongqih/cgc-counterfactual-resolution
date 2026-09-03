# Frozen-output context-anchoring contrasts and analytical figure

Capability: `FIGURE_FROM_FROZEN_PREDICTIONS`. Source: `358ca73bfd9eddb7afe3e02d01bce7ec12d39bc8`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py context_anchoring --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py context_anchoring --entrypoint scripts/cgc_context_anchoring.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `results/cgc_entrywise_compression/_cache/frozen_utility_table.npz` — ENTRYWISE_RUN_MANIFEST.json. Size: 27158852 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `results/cgc_entrywise_compression/_cache/bootstrap_draws.npz` — ENTRYWISE_RUN_MANIFEST.json. Size: 43200800 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- No raw reads, arrays, predictions or checkpoints are embedded. CLI smoke tests do not constitute a scientific rerun.
- Original figures are historical analytical outputs, not byte-exact reproduction of manually edited R2 SVGs.
- claim_boundary: The run command computes prespecified contrasts and emits Figure_Context_Anchoring PNG/SVG/source CSV from frozen utility/bootstrap tables. It does not refit predictors and does not reproduce manually edited R2 figures. Pass --authority-root explicitly.
