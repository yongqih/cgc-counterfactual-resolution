# Fixed PROGENy pathway-specific fidelity and recoverability diagnostic

## Execution

```text
python code/scientific/launch.py pathway_fidelity --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/progeny_pathway_fidelity_audit.py`

## Inputs

- `data/cgc_resolution2_replay/truth_g_primary_float32.npy`: Frozen Tahoe replicate truth
- `data/cgc_resolution2_replay/m49_k92_predictions_float32.npy`: Frozen all-but-one model predictions; no refit
- `data/cgc_resolution2_replay/m49_k92_frozen_weights.npz`: Frozen prediction replay episode weights
- `data/cgc_resolution2_replay/hierarchical_bootstrap_weights_int16.npy`: Frozen shared hierarchical bootstrap weights
- `results/cgc_entrywise_compression/_cache/frozen_utility_table.npz`: Frozen entrywise utility and support/sentinel assignment
- `data/cgc_bio1_official/frozen_program_weights.parquet`: Frozen PROGENy weights and gene identifiers
- `data/cgc_bio1_official/progeny`: Official PROGENy source revision cad6be0514c3248b9465e48f1cfd2f6a4c3dfb6f version 1.17.3
- `results/cgc_tahoe_0i/gene_metadata_frozen.csv`: Frozen gene metadata axis
- `results/cgc_tahoe_0i/gene_indices_g_primary.npy`: Frozen primary gene indices

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Recovery is undefined for nonpositive cross-replicate denominators, including VEGF.
- Supply --source-root explicitly.
