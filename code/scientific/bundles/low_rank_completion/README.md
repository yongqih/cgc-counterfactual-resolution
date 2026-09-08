# Additive versus CP low-rank interaction completion

Capability: `FULL_ANALYSIS`. Source: `d734df721a7914c965d99af1a6955cff6f64209c`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py low_rank_completion --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py low_rank_completion --entrypoint scripts/cgc_lowrank_run.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py low_rank_completion --entrypoint scripts/cgc_lowrank_masks.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py low_rank_completion --entrypoint scripts/cgc_lowrank_report.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/cgc_entrywise_cache` — LOW_RANK_INTERACTION_PROTOCOL.md; ENTRYWISE_FOUNDATION_MANIFEST.json. Size: not available bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- No raw reads, arrays, predictions or checkpoints are embedded. CLI smoke tests do not constitute a scientific rerun.
- Original figures are historical analytical outputs, not byte-exact reproduction of manually edited R2 SVGs.
