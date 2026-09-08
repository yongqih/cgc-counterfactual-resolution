# The recoverable resolution of cellular perturbation-response prediction

Analysis code accompanying *The recoverable resolution of cellular perturbation-response prediction*.

This repository contains 26 versioned scientific implementation bundles, their
configurations, split metadata, input requirements and execution documentation.
Figure source data and Supplementary Tables are distributed separately in the
companion data archive.

## 8 September 2026 correction

Hyperparameter selection is restricted to each observed support path, with sequence-aware downstream replay. Linked source data and Supplementary Table 5 are updated. The first detectable support budget, all-but-one point estimates and resolution conclusions remain unchanged. See [the correction guide](code/scientific/SUPPORT_BUDGET_CORRECTION.md). The previous v1.0.1 release is preserved.

## Start here

- [Code guide](CODE_README.md): implementation structure, environment and execution requirements.
- [Scientific entrypoint index](code/scientific/SCIENTIFIC_ENTRYPOINT_INDEX.json): available analyses and their execution classes.
- [Required inputs](REQUIRED_INPUT_DELIVERY.csv): supplied and external inputs for each analysis.
- [Dataset accessions](DATASET_ACCESSION_MANIFEST.csv): upstream datasets and resource versions.
- [Table checks and workbook generation](REPRODUCIBILITY_ENTRYPOINTS.md): checks on released source tables, distinct from scientific analysis runs.

From the repository root, inspect an analysis and its input requirements with:

```bash
python code/scientific/launch.py experimental_compression --inspect-only
```

This command does not download data or run an analysis. For scientific execution,
follow the relevant bundle README and input contract in
[`code/scientific/`](code/scientific/), together with the [code guide](CODE_README.md).
Large input arrays, predictions and model checkpoints are not included in this
repository; analysis-specific environment and historical provenance requirements
are documented in the code guide.

## Downloads and version

- **Manuscript code release:** [v1.0.1](https://github.com/yongqih/cgc-counterfactual-resolution/releases/tag/v1.0.1).
- **Prepared code archive:** [CGC_CODE_2026-09-08.zip](https://github.com/yongqih/cgc-counterfactual-resolution/releases/download/v1.0.1/CGC_CODE_2026-09-08.zip), attached to that release.
- **Companion source data:** [CGC_DATA_2026-09-08.zip](https://github.com/yongqih/cgc-counterfactual-resolution/releases/download/v1.0.1/CGC_DATA_2026-09-08.zip), attached to the same release.

If you clone this repository or use GitHub's **Code > Download ZIP**, the source
files are already provided; no second code ZIP is required.

The prepared code and data archives both use the top-level directory
`CGC_release_2026-09-08/`. Once both archives are available, extract them into the
same parent directory to combine their contents. For a GitHub checkout, place
the contents of the data archive's `CGC_release_2026-09-08/` directory at the
repository root. The data archive includes its own `DATA_README.md`.

The prepared code archive retains its archive-specific README. This repository
README provides GitHub-specific navigation; the corrected scientific implementations match the v1.0.1 archive. The original Zenodo deposits are retained; this patch does not replace them.

## License and attribution

CGC-authored code, configurations and documentation are distributed under the
[MIT License](LICENSE). Third-party resources retain their original terms; see
[License scope](LICENSE_SCOPE.md) and [Third-party notices](THIRD_PARTY_NOTICES.md).
