# Reproducing the analyses

## Prepared-data analyses

Run commands from the release root. Each computation writes to a separate workspace.

| Command after `python code/reproduce.py` | Computation | Required inputs |
| --- | --- | --- |
| `larry --work-dir ../reproduction` | Paired-error bounds for all 21 clones and three bootstrap sensitivity variants | Cell-level normalized RNA, cell metadata, gene identities, cohort and barcode provenance; included in DATA |
| `celltag-calibration --work-dir ../reproduction` | Calibration sets, held-out coverage, specificity and paired contrasts | 4,950 prediction rows, 150 split records and 165-clone labels; included in DATA |
| `celltag-fit --work-dir ../reproduction` | LR/RF nested model fitting with matched and shuffled information | Early RNA/ATAC features and six-fate labels; included in DATA; uses CPU parallelism |
| `tahoe-summary --work-dir ../reproduction` | Joint-resolution metrics and bootstrap inference | Prediction contributions, training-basis diagnostics and selection records; included in DATA |
| `tahoe-fit --work-dir ../reproduction --input-root PATH` | Joint-resolution model fitting | Prepared Gram caches at PATH; CUDA required |
| `tahoe-gene-selection --work-dir ../reproduction --input-root PATH` | Output-selection model fitting | Response tensors, gene indices and baseline cache at PATH; CUDA required |

`--inspect` describes a task without executing it. Data preparation scripts, accessions and checksums are included with the corresponding analysis.

## Scientific source bundles

`code/scientific/SCIENTIFIC_ENTRYPOINT_INDEX.json` maps 22 implementation bundles to the figures. `INPUT_CONTRACTS.json` specifies required inputs, and each `SOURCE_MANIFEST.json` records implementation provenance and file checksums.

```text
python code/scientific/launch.py experimental_compression --inspect-only
python code/scientific/launch.py lea_archs4 --inspect-only
```

Provide the listed inputs before executing an analysis. The T-cell prediction and decomposition stages require the recorded Git source objects and CUDA. Other bundles can require larger prepared arrays from the cited upstream studies. `reproducibility/INPUT_DELIVERY.csv` identifies these dependencies. To reproduce the figures from numerical tables, use the figure command below.

## Figures

```text
python code/render_figures.py --output ../rendered_figures
```

This renders all twelve figures from the source tables and three included SVG schematics. Arial was used for the supplied exports; a sans-serif fallback can change text metrics on systems without Arial.

`source_data/CURRENT_PANEL_MAP.csv` maps analysis tables to figure panels. Table identifiers are stable analysis identifiers; the panel map provides the display labels.

## Computational checks

`code/verify_release.py --code-only` checks a GitHub checkout. After extracting the DATA archive, `code/verify_release.py` checks the complete package. These commands verify SHA-256 integrity and figure coverage. `reproducibility/VALIDATION.json` records numerical recomputations, figure comparisons and software versions. Randomized analyses use the recorded seeds, splits and numerical precision.

### Implementation tests

With both archives extracted, run:

```text
python code/check_scientific.py --work-dir ../scientific-tests
```

The command runs tests for 18 scientific source bundles and the Tahoe joint-resolution analysis in separate workspaces. These tests cover training/test separation, algebraic identities, estimator behavior and selected reference results. The DATA archive includes the compact organoid and LCL reference tables required by three suites in `prepared/scientific_validation/`; its manifest records their source checksums. This command does not refit the full study. Use `--bundle NAME` to select one suite.
