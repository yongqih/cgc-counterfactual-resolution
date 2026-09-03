# Predefined five-pathway held-context scaling

Capability: `FULL_ANALYSIS`. Source: `22926886e3f87994492720ad2a529dd382596f09`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py five_pathway_scaling --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py five_pathway_scaling --entrypoint scripts/tahoe_five_pathway_scaling.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `results/cgc_tahoe_0i/response_tensors.zarr` — response_manifest.json. Size: not available bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_bio1_official/frozen_program_weights.parquet` — frozen pathway model source. Size: 2705687 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- No raw reads, arrays, predictions or checkpoints are embedded. CLI smoke tests do not constitute a scientific rerun.
- Original figures are historical analytical outputs, not byte-exact reproduction of manually edited R2 SVGs.
- claim_boundary: Externally predefined pathway set; no outcome-based selection. Historical uncertainty superseded by ed7_hierarchical_repair.
- active_execution_guard: Only the empirical prepare stage is allowlisted; original finalize/all produce superseded context-only intervals. Use ed7_hierarchical_repair for current inference.
