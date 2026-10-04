# The recoverable resolution of Virtual-Cell prediction

Analysis code, source data and figure-generation scripts for **The recoverable resolution of Virtual-Cell prediction**.

The study examines which biological detail can be predicted from a specified target, the information available at prediction time and a required level of reliability. Analyses cover T-cell perturbations, State predictions, Tahoe experimental support and response resolution, LCL response programs, organoid drug ranking, LARRY lineages and CellTag-multi fate prediction.

## Getting started

The GitHub repository contains code and figure-level source tables. Figure rendering requires this repository alone. The CODE and DATA archives share a `CGC/` directory; extract them into the same location to use the prepared numerical inputs. On Windows, use a short extraction path.

The prepared-data analyses were tested with Python 3.10.20. Package versions and computational checks are recorded in `reproducibility/VALIDATION.json`.

```text
python -m pip install -r requirements.txt
python code/verify_release.py --code-only
python code/render_figures.py --output ../rendered_figures
```

With the DATA archive extracted:

```text
python code/verify_release.py
python code/reproduce.py larry --work-dir ../reproduction
python code/reproduce.py celltag-calibration --work-dir ../reproduction
python code/reproduce.py tahoe-summary --work-dir ../reproduction
```

`REPRODUCIBILITY.md` describes each computation, including the inputs and hardware required for model fitting. Commands write to separate output directories.

## Contents

| Directory | Contents |
| --- | --- |
| `code/figures/` | Renderers for five main figures, three Extended Data figures and four supplementary figures |
| `code/analysis/` | LARRY, CellTag-multi and Tahoe resolution analyses |
| `code/scientific/` | T-cell, State, LCL, organoid and perturbation-response implementations |
| `source_data/` | Numerical figure tables, pathway weights and panel mapping |
| `prepared/` | Processed measurements, predictions, calibration records and sufficient statistics in the DATA archive |
| `figures/` | PDF, SVG, PNG and TIFF figure exports in the DATA archive |
| `manuscript/` | Main article and Supplementary Information in the DATA archive |
| `reproducibility/` | Input specifications, software versions and computational checks |

Repository: [cgc-counterfactual-resolution](https://github.com/yongqih/cgc-counterfactual-resolution). Upstream datasets are listed in `DATASET_ACCESSION_MANIFEST.csv`; input requirements are listed in `reproducibility/INPUT_DELIVERY.csv`.

CGC-authored code is MIT licensed. Third-party data and software retain their providers' terms; see `THIRD_PARTY_NOTICES.md`.
