# The recoverable resolution of Virtual-Cell prediction

Code and data for the author-approved manuscript of 4 October 2026: **five main figures, three Extended Data figures and four supplementary figures**.

The experiments assess which biological detail is supported by the target, information available at prediction time and required reliability. The release includes T-cell prediction, State artifact evaluation, Tahoe experimental support and response resolution, LCL response programs, patient-derived organoid drug ranking, LARRY split-culture lineages and CellTag-multi fate prediction.

## Start here

Extract the CODE and DATA archives into the same parent directory. Both contain `CGC/` and merge into one release. On Windows, use a short extraction path. The prepared-data analyses were checked with Python 3.10.20; exact package versions are recorded in `reproducibility/VALIDATION.json`. Install the requirements in a dedicated environment.

```text
python -m pip install -r requirements.txt
python code/verify_release.py
python code/render_figures.py --output ../rendered_figures
python code/reproduce.py larry --work-dir ../reproduction
python code/reproduce.py celltag-calibration --work-dir ../reproduction
python code/reproduce.py tahoe-summary --work-dir ../reproduction
```

`REPRODUCIBILITY.md` distinguishes full recomputation, analysis from frozen predictions, figure rendering and file-integrity checks. It lists the additional inputs and compute needed for refitting earlier models. Approved manuscript files and figures are read-only references; commands write to separate directories.

## Layout

| Directory | Contents |
| --- | --- |
| `code/figures/` | Portable renderers and the author's preserved SVG schematics |
| `code/analysis/` | Current LARRY, CellTag and Tahoe analysis code and protocols |
| `code/scientific/` | Twenty-two frozen scientific bundles supporting the retained experiments |
| `source_data/` | Current figure-level numerical tables and mappings |
| `prepared/` | Compact measured inputs, predictions, calibration records and sufficient statistics |
| `figures/` | Approved PDF, SVG, PNG and TIFF figure exports |
| `manuscript/` | Approved main and supplementary documents and figure review PDFs |
| `reproducibility/` | Source provenance, input delivery and validation records |

`MANUSCRIPT_AUTHORITY.json` identifies this revision. The repository is [cgc-counterfactual-resolution](https://github.com/yongqih/cgc-counterfactual-resolution). The earlier archived release is v1.0.1, DOI [10.5281/zenodo.22664234](https://doi.org/10.5281/zenodo.22664234). Updated CODE and DATA archives have been prepared for this revision; their Zenodo deposition and new archival identifier are pending.

CGC-authored code is MIT licensed. Dataset and third-party terms remain with their respective providers; see `THIRD_PARTY_NOTICES.md` and `DATASET_ACCESSION_MANIFEST.csv`.
