# Transfer of the fold-local full training-response span

## Execution

```text
python code/scientific/launch.py lea_full_span --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/run_full_span_pca_sanity_audit.py`

## Inputs

- `data/wgcna_frontier_official/MATCHED_POSTSVA_340_REPRESENTATION.npz`: Matched Lea 340-LCL response representation and fixed folds
- `data/wgcna_frontier_official/MATCHED_POSTSVA_FULL_OOF.npz`: Matched post-SVA six-model OOF archive
- `results/level2_lea/archs4_robustness/ARCHS4_GENE_AXIS_RECONCILIATION.csv`: Frozen matched gene axis
- `results/cgc_dimension_matched_compression`: Historical PCA curve/summary comparison required by reporting

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Fits PCA within each training fold and measures transfer of the response span.
- run_wgcna_resolution_frontier.py supplies data-loading functions for this calculation.
