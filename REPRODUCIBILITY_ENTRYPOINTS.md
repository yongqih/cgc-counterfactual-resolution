# Release integrity checks and workbook generation

Audit status: these 17 Python wrappers are **not analysis or figure reproductions**. They read plotting/summary CSVs, not the upstream biological inputs listed below. The separate JavaScript workbook builder is an executable TABLE_GENERATOR. The current manuscript is Main 1–6 / ED1–9 / SF1–5, and all 20 corresponding released source-data tables are present.

Run commands from the release root. Each command writes a compact JSON summary to the legacy path `reproduced/` and verifies selected manuscript-facing values or row counts against released source tables. A successful summary explicitly reports `integrity_check_passed` and `scientific_reproduction: false`.

## Primary T-cell held-donor prediction

- Upstream study/resource: Primary human CD4+ T-cell Perturb-seq
- Command: `python code/tcell_cgc/run.py`
- Expected output: `reproduced/tcell_cgc_summary.json`
- Consumed by: Main Figure 1; Supplementary Figure 1
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## State held-context cytokine prediction

- Upstream study/resource: State cytokine prediction artifacts
- Command: `python code/state_held_context/run.py`
- Expected output: `reproduced/state_held_context_summary.json`
- Consumed by: Main Figure 2
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Independent State Tahoe held-context validation

- Upstream study/resource: State Tahoe prediction artifacts
- Command: `python code/state_tahoe_validation/run.py`
- Expected output: `reproduced/state_tahoe_validation_summary.json`
- Consumed by: Extended Data Figure 1
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Tahoe replicate-core construction

- Upstream study/resource: Tahoe-100M
- Command: `python code/tahoe_replicate_core/run.py`
- Expected output: `reproduced/tahoe_replicate_core_summary.json`
- Consumed by: Methods; Supplementary Tables 1–3
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Strict Experimental Compression

- Upstream study/resource: Tahoe-100M
- Command: `python code/experimental_compression/run.py`
- Expected output: `reproduced/experimental_compression_summary.json`
- Consumed by: Main Figure 3; Extended Data Figure 2; Supplementary Table 5
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Generic context calibration

- Upstream study/resource: Tahoe-100M
- Command: `python code/generic_context_calibration/run.py`
- Expected output: `reproduced/generic_context_calibration_summary.json`
- Consumed by: Supplementary Figure 2
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Low-rank context-by-intervention completion

- Upstream study/resource: Tahoe-100M
- Command: `python code/low_rank_completion/run.py`
- Expected output: `reproduced/low_rank_completion_summary.json`
- Consumed by: Main Figure 3e; Extended Data Figure 3
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Same-context anchoring

- Upstream study/resource: Tahoe-100M
- Command: `python code/context_anchoring/run.py`
- Expected output: `reproduced/context_anchoring_summary.json`
- Consumed by: Extended Data Figure 4
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Lea RNA-only counterfactual prediction

- Upstream study/resource: Lea GSE207049
- Command: `python code/lea_rna_only/run.py`
- Expected output: `reproduced/lea_rna_only_summary.json`
- Consumed by: Main Figure 4
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Lea ARCHS4 robustness

- Upstream study/resource: ARCHS4 v2.5 and Lea GSE207049
- Command: `python code/lea_archs4/run.py`
- Expected output: `reproduced/lea_archs4_summary.json`
- Consumed by: Extended Data Figure 5
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Lea training-response span transfer

- Upstream study/resource: Lea GSE207049
- Command: `python code/lea_span_transfer/run.py`
- Expected output: `reproduced/lea_span_transfer_summary.json`
- Consumed by: Main Figure 4e; Extended Data Figure 5
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Tahoe operator anatomy

- Upstream study/resource: Tahoe-100M
- Command: `python code/tahoe_operator_anatomy/run.py`
- Expected output: `reproduced/tahoe_operator_anatomy_summary.json`
- Consumed by: Main Figure 5
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Mechanistic attribution of the unresolved operator

- Upstream study/resource: Tahoe-100M; HGNC; PROGENy
- Command: `python code/mechanistic_attribution/run.py`
- Expected output: `reproduced/mechanistic_attribution_summary.json`
- Consumed by: Supplementary Figure 3
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Mechanism-aligned multimodal prediction

- Upstream study/resource: Tahoe-100M; DepMap; MCLP RPPA500; Sanger proteomics; Reactome
- Command: `python code/multimodal_prediction/run.py`
- Expected output: `reproduced/multimodal_prediction_summary.json`
- Consumed by: Supplementary Figure 4; Supplementary Table 6
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Counterfactual Resolution analysis

- Upstream study/resource: Tahoe-100M; PROGENy
- Command: `python code/counterfactual_resolution/run.py`
- Expected output: `reproduced/counterfactual_resolution_summary.json`
- Consumed by: Main Figure 6; Extended Data Figure 6; Supplementary Table 4
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Figure source-data assembly

- Upstream study/resource: Released manuscript-facing source-data tables
- Command: `python code/figure_generation/run.py`
- Expected output: `reproduced/figure_generation_summary.json`
- Consumed by: Main 1–6; Extended Data 1–9; Supplementary 1–5
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Supplementary Table assembly

- Upstream study/resource: Released supplementary-table source CSVs
- Command: `python code/supplementary_tables/run.py`
- Expected output: `reproduced/supplementary_tables_summary.json`
- Consumed by: Supplementary Tables 1–6
- Implemented level: `NON_REPRODUCTION_INTEGRITY_CHECK`
- Full reconstruction: not implemented by this wrapper.

## Actual Supplementary Table generation

- Capability: `TABLE_GENERATOR`
- Command: `node --max-old-space-size=10240 code/supplementary_tables/build_workbooks.mjs`
- Inputs: six CSVs in `supplementary_table_sources/`
- Outputs: six XLSX workbooks and previews; Table5 budget columns are formulas.
- See `code/supplementary_tables/README.md` for staging outputs and dependencies.

The intended scientific categories FULL_ANALYSIS, FIGURE_FROM_FROZEN_PREDICTIONS and OFFICIAL_ARTIFACT_SUMMARY must not be claimed until corresponding portable code is actually packaged and tested. Author-edited SVGs are preserved; no figure is regenerated by these checks.

## Actual scientific implementations (release repair)

The wrapper descriptions above are deliberately unchanged: they remain checks. Actual scientific implementations now reside in `code/scientific/`, with byte-exact source/configuration/split metadata and an executable source exporter. See [scientific README](code/scientific/README.md), `SCIENTIFIC_ENTRYPOINT_INDEX.json`, and `INPUT_CONTRACTS.json`. The classes and runtime requirements are distinct. Current State and Resolution2 entrypoints are guarded against superseded inference; ED7 uses the hierarchical repair. T-cell1B/1D require original historical Git provenance and CUDA, not just the code ZIP. No scientific stage was run in this repair.
