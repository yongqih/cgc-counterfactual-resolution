# Hierarchical uncertainty for held-context response recovery

## Execution

```text
python code/scientific/launch.py ed7_hierarchical_repair --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:


## Inputs

- `.worktrees/tahoe_held_context_scaling/results/tahoe_held_context_scaling`: abb503e9f3cf75eaea7a075f81160db3d04de64d fixed Grams, hyperparameters and point estimates
- `.worktrees/tahoe_five_pathway_scaling/results/tahoe_five_pathway_scaling/_cache/TAHOE_FIVE_PATHWAY_SCALING_CONTRIBUTIONS.npz`: 22926886e3f87994492720ad2a529dd382596f09

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Computes context-then-intervention bootstrap intervals from saved prediction contributions.
- The entrypoint has no command-line help parser; inspect requirements through the launcher.
