# Tahoe full-transcriptome replicate geometry and operator anatomy

Capability: `FULL_ANALYSIS`. Source: `c96ee83bd59e3974bc42cd1cfc2aecf7b7dd9e70`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py tahoe_operator_anatomy --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py tahoe_operator_anatomy --entrypoint scripts/cgc_tahoe_0i_scaling.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `results/cgc_tahoe_0i/response_tensors.zarr` — response_manifest.json. Size: not available bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `results/cgc_tahoe_0i/gene_indices_g_primary.npy` — gene_universe_hashes.json. Size: 102908 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- No raw reads, arrays, predictions or checkpoints are embedded. CLI smoke tests do not constitute a scientific rerun.
- Original figures are historical analytical outputs, not byte-exact reproduction of manually edited R2 SVGs.
- claim_boundary: q_N is response-observed same-intervention support fidelity, not target-blind predictive recovery. R_dim is cross-plate remaining-residual energy. Original per-stage scripts resolve paths from their bundle root.
