# Strict target-hidden entrywise Experimental Compression

Capability: `FULL_ANALYSIS`. Source: `a893cc338c11f402a9ef9966bd376f2aca92cb88`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py experimental_compression --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_foundation.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_surface.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_controls.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_inference.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_report.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/cgc_entrywise_cache/same_plate6_float32.npy` — ENTRYWISE_FOUNDATION_MANIFEST.json. Size: 86490128 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_entrywise_cache/same_plate14_float32.npy` — ENTRYWISE_FOUNDATION_MANIFEST.json. Size: 86490128 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_entrywise_cache/cross_plate6_plate14_float32.npy` — ENTRYWISE_FOUNDATION_MANIFEST.json. Size: 86490128 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_entrywise_cache/sample_gene_sums_float64.npy` — ENTRYWISE_FOUNDATION_MANIFEST.json. Size: 74528 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- No raw reads, arrays, predictions or checkpoints are embedded. CLI smoke tests do not constitute a scientific rerun.
- Original figures are historical analytical outputs, not byte-exact reproduction of manually edited R2 SVGs.
