# Final BIO2 attribution and MULTI2 mechanism-aligned prediction

Capability: `FULL_ANALYSIS`. Source: `0c8eeb48fbbf09b645e4e792bfe205158ca86ce3`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py mechanism_aligned --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points

- `python code/scientific/launch.py mechanism_aligned --entrypoint scripts/cgc_mechanism_bio2.py --execute -- <original CLI arguments; see safe help>`
- `python code/scientific/launch.py mechanism_aligned --entrypoint scripts/cgc_mechanism_multi2.py --execute -- <original CLI arguments; see safe help>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `data/cgc_bio1_official/bio1_gene_arrays.npz` — MULTI2 SOURCE_HASHES; BIO2 EXPECTED_GENE_HASH. Size: 1911708512 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_bio1_official/bio1_projected_programs.npz` — BIO2 EXPECTED_PROGRAM_HASH. Size: 191722116 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_multi1_official/rppa500-mclp_tahoe49.parquet` — MULTI2 SOURCE_HASHES. Size: 210556 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_multi1_official/harmonized_Sanger_MS_2022_tahoe49.parquet` — MULTI2 SOURCE_HASHES. Size: 2331840 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_multi1_official/Chronos_Combined_tahoe49.parquet` — MULTI2 SOURCE_HASHES. Size: 7068304 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_mechanism_followup` — MULTI2_ANNOTATION_PROVENANCE.json. Size: not available bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `results/cgc_tahoe_0i/response_tensors.zarr` — MULTI2 load_foundation baseline-only primary_dmso_cpm. Size: not available bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `results/cgc_tahoe_0i/gene_indices_g_primary.npy` — gene_universe_hashes.json. Size: 102908 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `data/cgc_bio1_official/frozen_program_weights.parquet` — MULTI2 SOURCE_HASHES. Size: 2705687 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- No raw reads, arrays, predictions or checkpoints are embedded. CLI smoke tests do not constitute a scientific rerun.
- Original figures are historical analytical outputs, not byte-exact reproduction of manually edited R2 SVGs.
- claim_boundary: Power denotes detection under oracle-aligned injected delta-g, not an unbiased fitted-effect MDE. Dependency is an oracle input, not deployment-available untreated biology.
- path_adapter: bio2: launcher sets only common/bio2 ROOT, SOURCE_ROOT and OUT before invoking unchanged run_bio2; multi2 accepts --root --source-root --out.
- BIO2 portable path command: `python code/scientific/launch.py mechanism_aligned --entrypoint scripts/cgc_mechanism_bio2.py --input-root <prepared-input-root> --work-root <prepared-metadata-output-workspace> --execute -- --formal`. The adapter changes ROOT/SOURCE_ROOT/OUT only.
- MULTI2 uses its original `--root`, `--source-root`, `--out`, `--formal` flags. It can be expensive; a CLI smoke test is not a power/run/null execution.
