# Frozen-contribution ED7 hierarchical uncertainty repair

Capability: `FULL_ANALYSIS`. Source: `c2ffd06e44be10c77f3b68dbae880ccdf22f6f1f`.

Runtime: **PREPARED_INPUTS_REQUIRED; scientific run not performed**.

The files listed in SOURCE_MANIFEST.json are byte-exact historical Git blobs. The launcher and any current bounded release reporting entry are separate, explicitly identified release code. No frozen scientific source was rewritten.

## Safe inspection

```text
python code/scientific/launch.py ed7_hierarchical_repair --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. No stage was run during this release repair.

## Entry points


For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `.worktrees/tahoe_held_context_scaling/results/tahoe_held_context_scaling` — abb503e9f3cf75eaea7a075f81160db3d04de64d frozen Grams, hyperparameters and point estimates. Size: not available bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.
- `.worktrees/tahoe_five_pathway_scaling/results/tahoe_five_pathway_scaling/_cache/TAHOE_FIVE_PATHWAY_SCALING_CONTRIBUTIONS.npz` — 22926886e3f87994492720ad2a529dd382596f09. Size: 28793698 bytes. not copied into code bundle; see DATA_AVAILABILITY_FINAL.md.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- No raw reads, arrays, predictions or checkpoints are embedded. CLI smoke tests do not constitute a scientific rerun.
- Original figures are historical analytical outputs, not byte-exact reproduction of manually edited R2 SVGs.
- claim_boundary: Only frozen-contribution uncertainty; unchanged predictors, point definitions and draws. No CLI --help is advertised because the original script has no argument parser.
- path_adapter: ed7: launcher resolves GENE_WT/PATH_WT to exported frozen bundles or explicit prepared workspaces and patches directory constants only before unchanged main.
- Explicit command: `python code/scientific/launch.py ed7_hierarchical_repair --execute --work-root <new-audit-workspace> --gene-workspace <prepared-held-context-workspace> --pathway-workspace <prepared-pathway-workspace>`. Only directory constants are rebound; the frozen hierarchical bootstrap is unchanged.
