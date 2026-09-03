# Target-hidden Tahoe context-support scaling

Capability: `FULL_ANALYSIS`. Source: `abb503e9f3cf75eaea7a075f81160db3d04de64d`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py held_context_scaling --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py held_context_scaling --entrypoint scripts/tahoe_held_context_scaling.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py held_context_scaling --entrypoint scripts/tahoe_held_context_secondary.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `results/cgc_tahoe_0i/response_tensors.zarr` — response_manifest.json. Size: not available bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/tahoe100m_control_state/control_wells_log1p_cpm_float32.npy` — frozen control-state source manifest. Size: 34018528 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `results/cgc_tahoe_0d/lineage_baseline.csv` — frozen lineage metadata. Size: 11137 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- No raw reads, arrays, predictions or checkpoints are embedded. CLI smoke tests do not constitute a scientific rerun.
- Original figures are historical analytical outputs, not byte-exact reproduction of manually edited R2 SVGs.
- claim_boundary: Historical finalize uses context-only uncertainty and is NOT final ED7 inference. The separately packaged ed7_hierarchical_repair must supply current context-then-intervention uncertainty. Point estimates/models remain frozen.
- active_execution_guard: Old context-only finalizer is not allowlisted. Use ed7_hierarchical_repair for current inference.
