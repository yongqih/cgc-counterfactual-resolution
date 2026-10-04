# Primary CD4 T-cell pseudobulk truth construction and guide-disjoint reliability

## Execution

```text
python code/scientific/launch.py tcell_truth --inspect-only
```

`--inspect-only` lists requirements. Use `--cli-help` for supported parsers and `--execute` after preparing the inputs.

Entrypoints:

- `src/igc_virtual_cell/cgc_tcell/prepare.py`
- `src/igc_virtual_cell/cgc_tcell/guide_disjoint.py`
- `src/igc_virtual_cell/cgc_tcell/analysis.py`

## Inputs

- `data/raw/cgc_tcell/GWCD4i.pseudobulk_merged.h5ad`: Official processed guide x donor x culture pseudobulk; exact URL and checksum in download_manifest.json

Paths are relative to the supplied workspace. `SOURCE_MANIFEST.json` records provenance and file checksums.

## Analysis and runtime requirements

- Reconstructs guide-disjoint donor-state response contrasts from official pseudobulk data.
