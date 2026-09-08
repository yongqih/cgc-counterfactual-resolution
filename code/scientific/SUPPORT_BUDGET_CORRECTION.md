# Support-budget correction — 8 September 2026

This patch corrects hyperparameter selection in the entrywise Experimental
Compression grid: each support path now uses only the response-observed reference
contexts available in that path. Tuning can pool paths only when their reference
sets are identical. Downstream replay preserves the support-sequence parameter.
Old and corrected caches cannot be mixed. Historical source commits identify the
base implementation; SOURCE_MANIFEST.json explicitly records corrected files.

The full 50-target, 90-budget grid and its original 10,000 hierarchical bootstrap
draws were recomputed, followed by the linked correspondence-null summaries,
paired anchoring contrasts and retained gene/pathway comparisons. Datasets,
splits, models, seeds, metrics and multiplicity families were not changed.

## Corrected results

| Endpoint | Corrected value |
|---|---:|
| First detectable entrywise recovery, m=40, k=4 | 0.08029583204114876 |
| Its 90-grid simultaneous lower bound | 0.00688087637144813 |
| First detectable measured support | 3724/4650 = 80.09% |
| All-but-one recovery, m=49, k=92 | 0.11068053088808294 (unchanged) |
| All-but-one simultaneous lower bound | 0.04551831993351292 |
| Anchoring gain, m=40, k=2 | 0.07667128657442189 |
| Paired 12-contrast interval for that gain | [-0.000560953056852373, 0.15390352620569614] |
| All-but-one pathway recovery | 0.27281393788398656 (unchanged to numerical precision) |
| Pathway-minus-gene interval at all-but-one | [0.07441147818369379, 0.24985533580811345] |

The two-anchor lower bound no longer exceeds zero; the broad-reference,
front-loaded anchoring pattern remains. The first-detection budget, all-but-one
point estimates and resolution conclusions remain unchanged. Fig. 3, ED2d,
ED4, the Main Fig. 6 interval and Supplementary Table 5 are synchronized.
ED7 is a distinct held-context scaling task and was not changed. Lea, T-cell,
low-rank completion and organoid analyses were not rerun in this patch.

## Execution

Use a prepared workspace ROOT with the input paths in INPUT_CONTRACTS.json.
Preserve the original foundation and split manifests. In a new prepared output
workspace (or after separately archiving old outputs), run these existing stages:

```text
python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_surface.py --execute -- run --root ROOT --target-start 0 --target-stop 50
python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_surface.py --execute -- aggregate --root ROOT
python code/scientific/launch.py experimental_compression --entrypoint scripts/cgc_entrywise_inference.py --execute -- --root ROOT
```

The included experimental-compression result directory contains the corrected
surface, frontier, thresholds, intervals, null summaries and hashed correction
manifest. Large per-target caches, expression arrays and prediction tensors are
external, as in v1.0.0. The manifest can identify the archived corrected caches;
it is not a substitute for those inputs or a promise of bitwise equivalence
across numerical environments.

For downstream reconstruction, use `ReplayContext(ROOT, ROOT)` and
`materialize_predictions` from the resolution bundle's `cgc_resolution_poc.replay`.
The latter accepts `budgets=((40, 4, 8),)` to rebuild only the affected tensor;
the all-but-one tensor is unchanged. Supply the manifest, original truth and
split metadata, corrected per-target caches, and required Gram arrays.

`recompute_corrected_pathway.py --help` documents the bounded paired-inference
entrypoint. It requires the corrected replay, original pathway weights,
bootstrap weights, frozen calibration membership table and candidate-score
arrays. It returns only the two retained pathway-minus-gene comparisons plus
their four point estimates. The original four-term max-T calibration is kept;
random-panel terms do not regain a biological-control interpretation.
For figure-only use, the default resolution renderer reads the included
`RESOLUTION2_VALID_COMPARISONS_CORRECTED.csv`; historical tables are superseded.

The 2026-09-08 tests cover support restriction, identical-set pooling, cache
versioning, sequence-aware replay, corrected-row selection and release guards.
These tests supplement, and do not replace, the real-data rerun described above.
No automated script reproduces the author's Illustrator layout byte-for-byte.
