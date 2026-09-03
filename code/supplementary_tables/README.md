# Supplementary Table generator

Capability: **TABLE_GENERATOR**. `build_workbooks.mjs` is the actual workbook generator, adapted from the previously used Artifact Tool generator. `run.py` is only a source-CSV smoke check; it does not generate workbooks.

Requires Node.js and `@oai/artifact-tool` in the module resolution path. Run from any directory. Numeric arguments select table numbers; omitting them selects all six. The large 202,302-row Table 4 requires more than Node's default 4 GB heap on this host.

```text
node --max-old-space-size=10240 publication_release/code/supplementary_tables/build_workbooks.mjs 1 2 3 4 5 6
```

Default output is `publication_release/`; set `CGC_TABLE_OUTPUT_DIR` and `CGC_TABLE_PREVIEW_DIR` to audited staging directories to avoid overwriting release files during validation. Inputs are the six frozen/metadata-reconciled CSVs in `supplementary_table_sources/`. No scientific analysis is invoked.

Table 5 contains spreadsheet formulas for measured-entry count and fraction, with the 93-intervention and 4,650-entry constants on its Notes sheet. Table 4 uses the complete fixed weight CSV and a bounded preview to keep visual QA manageable. Validate every exported Data cell against its source CSV; a preview is not a substitute for full-cell reconciliation.

Original workbook/source backups for the 2026-09-03 audit are in `audit/final/release_before/`. Provenance and claim boundaries are in the audit reports. The generator preserves the established table presentation and adds column widths for audit metadata; it never edits manuscript figures.
