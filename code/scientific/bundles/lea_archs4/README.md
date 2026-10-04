# Independent ARCHS4 pre-SVA quantification robustness

## Execution

```text
python code/scientific/launch.py lea_archs4 --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `scripts/run_lea_archs4_robustness.py`
- `scripts/prepare_lea_archs4_robustness.py`

## Inputs

- `data/level2_lea_archs4_matched`: ARCHS4 v2.5 matched count extraction and prepared matched 340-LCL/10110-gene representations; exact hashes in extraction/run manifests
- `results/level2_lea/LEVEL2_OOF_FROZEN.npz`: Historical Lea fixed 342-LCL OOF and exact matched-core selection
- `data/level2_lea_official/GSE207049_31Mar21_all_runs_voom_resid.txt.gz`: Matched post-SVA comparator from author GEO matrix

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Execution loads the recorded c3d074 source objects; lea_historical_analysis provides the corresponding local imports.
- Supply an explicit source root with the documented directory layout.
- ARCHS4 provides Kallisto-derived counts; the Lea study provides HTSeq-derived data and a post-SVA representation.
- Extraction uses HTTP range requests.
