# Reproducing the current manuscript

## Current analysis entrypoints

Run from the release root after extracting both archives. Each command preserves the approved source data and writes to a separate workspace.

| Command after `python code/reproduce.py` | Computation | Inputs delivered |
| --- | --- | --- |
| `larry --work-dir ../reproduction` | All 21 clones, paired-error floors and three bootstrap sensitivity variants | Cell-level normalized RNA, metadata, gene identities, cohort and barcode provenance |
| `celltag-calibration --work-dir ../reproduction` | Probabilities, 150 disjoint splits, calibration sets, held-out coverage, specificity and paired contrasts | All 4,950 prediction rows, fitting records, 165-clone labels and early feature matrix |
| `celltag-fit --work-dir ../reproduction` | LR/RF nested model fitting with matched and shuffled information | Prepared early RNA/ATAC matrix and six-fate labels; uses CPU parallelism |
| `tahoe-summary --work-dir ../reproduction` | Joint-resolution metrics and bootstrap inference | Frozen prediction contributions, train-only basis diagnostics and selection records |
| `tahoe-fit --work-dir ../reproduction --input-root PATH` | Refit the original joint-resolution experiment | Additional prepared Gram caches at PATH; CUDA required |
| `tahoe-gene-selection --work-dir ../reproduction --input-root PATH` | Refit the output-selection controls | Original response tensors, gene indices and baseline cache at PATH; CUDA required |

`--inspect` describes a task without executing it. Compact prepared inputs allow the first four routes to run without downloading full source archives. Original download and feature-preparation code is included with its accessions and checksums.

## Earlier experiments retained in the manuscript

`code/scientific/SCIENTIFIC_ENTRYPOINT_INDEX.json` maps 22 source bundles to the current figures. `INPUT_CONTRACTS.json` records their required prepared inputs; each bundle's `SOURCE_MANIFEST.json` fixes its scientific sources and historical provenance.

```text
python code/scientific/launch.py experimental_compression --inspect-only
python code/scientific/launch.py lea_archs4 --inspect-only
```

These bundles retain their original execution contracts. In particular, the T-cell prediction and decomposition stages require original historical Git provenance and CUDA, and several earlier analyses require larger prepared inputs available from the cited upstream resources. Their source code is delivered, but the compact archive is not a replacement for those inputs. Earlier artifact summaries and corrected pathway contrasts remain distinct from refitting models.

## Figures and source tables

```text
python code/render_figures.py --output ../rendered_figures
```

This renders all twelve final figures using only released source tables and three preserved author SVG panels. Figure rendering does not refit models. Arial was used for approved exports; a sans-serif fallback can change text metrics on systems without Arial. The approved exports remain the visual reference.

All current panels are listed in `source_data/CURRENT_PANEL_MAP.csv`. Legacy internal figure labels in numerical tables are retained as provenance; current panel labels are authoritative.

## Interpretation of verification

`code/verify_release.py` verifies SHA-256 file integrity and figure coverage. `reproducibility/VALIDATION.json` records the separate numerical recomputations and figure comparisons performed for this release. Randomized analyses retain their recorded seeds, splits and original numerical precision.
