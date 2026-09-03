# Code archive

## What is supplied

`code/scientific/` contains 26 frozen implementation bundles, configurations,
split metadata, input contracts and the tested environment.
`SCIENTIFIC_ENTRYPOINT_INDEX.json` maps them to manuscript analyses. Classes
describe the implemented operation with its required inputs: FULL_ANALYSIS,
FIGURE_FROM_FROZEN_PREDICTIONS, or OFFICIAL_ARTIFACT_SUMMARY. They do not imply
that the code ZIP includes all data or all historical execution prerequisites.

The 17 lightweight `code/<task>/run.py` wrappers are explicitly labeled
NON_REPRODUCTION_INTEGRITY_CHECK. They check delivered tables, not fit models.
The workbook builder is a separate TABLE_GENERATOR; delivered CSVs and XLSX
files can be read without its authoring environment.

## Safe, locally testable entrypoints

From the combined extracted release directory:

```
python code/scientific/launch.py experimental_compression --inspect-only
python code/figure_generation/run.py --output checks/figures.json
python code/supplementary_tables/run.py --output checks/tables.json
```

The first command inspects execution requirements; the final two also need
the data ZIP. No command above downloads input or reruns scientific outcomes.
Actual analyses require an explicit `--execute`; consult each bundle README,
`SOURCE_MANIFEST.json`, input contracts and the original CLI before using it.

## Required external environment and inputs

Large expression/prepared arrays, fitted predictions and checkpoints remain
external, with upstream accession and frozen input contracts recorded. The
recorded environment is an author-tested environment, not a cross-platform
lockfile. Optional model packages, CUDA and resource-specific terms still apply.
Historical absolute paths in immutable source blobs are provenance, not portable
defaults; use documented path options/adapters where supplied.

`REQUIRED_INPUT_DELIVERY.csv` marks the delivery status of all 85 recorded input
contracts. A historical `local_presence` flag means presence in the authors'
workspace, not presence in either uploaded archive or a public download link.

The T-cell 1B (`tcell_lodo`) and 1D (`tcell_decomposition`) source enforces genuine
original-branch/history and unchanged-artifact gates as well as CUDA. The code
ZIP does not provide that original Git history, and no public remote for it was
configured at packaging time. Those two bundles are source-available archival
implementations, not standalone reproducible ZIP entrypoints. Their provenance
gates were not bypassed and no fake history was created. The same limitation
does not turn table-integrity checks into scientific reruns.

The historical Experimental Compression reporting stage also invokes an
author-local Python executable inside `_invariance_report`; it is preserved
as source rather than advertised as a portable report command. Its Python
path must be adapted explicitly before use on a different computer. Source
inspection and `--inspect-only` do not validate that reporting-stage runtime.

The formatted workbook generator uses Node.js and `@oai/artifact-tool`, whose
public installation is not assumed; the six ready-made XLSX and CSV tables are
in the data archive. Analytical renderers do not reconstruct author-edited SVG
artwork byte-for-byte. No artwork or historical figure versions are modified.

The released State, Resolution and ED7 adapters preserve the approved inference
boundaries. Historical all-stages code is archival where explicitly labeled;
do not select obsolete inference as a current manuscript result.

## Attribution

CGC-authored code and documentation are MIT licensed. Third-party resources
remain under their original terms. See LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md.
The archive contains no virtual environments, node_modules, Git repository,
download credentials or model binaries. Scientific implementations are unchanged.
