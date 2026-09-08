# Executable scientific source package

This package contains versioned scientific implementations, not renamed integrity checks. Its separate 17 lightweight release wrappers remain NON_REPRODUCTION_INTEGRITY_CHECK utilities. The scientific capability classes below describe what code does when supplied with its required inputs; they do not assert that raw/prepared data or all historical worktrees are embedded.

Current export: 26 bundles, 500 registered source/configuration/result files. Historical versions and corrected bytes are distinguished in SOURCE_MANIFEST.json. Large expression matrices, prediction arrays and checkpoints remain external.

## Start here

1. Read SCIENTIFIC_ENTRYPOINT_INDEX.json and the relevant bundle README.
2. Use `python code/scientific/launch.py <bundle> --inspect-only`; this does not run an analysis.
3. Inspect safe help with `--cli-help`, then stage the exact inputs and environment. Require an explicit `--execute` for scientific work.
4. Validate source hashes with `python code/scientific/export_frozen_sources.py --verify-only`.

Corrected bundles cannot be recreated from historical Git blobs alone; the exporter rejects that operation. For unchanged bundles, `export_frozen_sources.py --repo <repository-with-recorded-commits>` reads Git blobs. It never imports scientific code, downloads data or fits models. The included files are already exported; the export utility is not a substitute for them.

## Active versus archival execution

- State discovery: the active bounded official-artifact summary uses the unchanged official loader/metrics and the audited exact 44 unique five-context derangements with conservative plus-one p. It cannot invoke the obsolete sampled-null main. It does not claim to regenerate bootstrap intervals or State predictions.
- Resolution-v2: the active renderer consumes only the six valid frozen gene/pathway points/contrasts across the two budgets. It cannot launch or display the invalid random-projection arm. It creates an analytical figure, not the manual R2 SVG layout.
- ED7: historical context-only finalizers are not allowlisted. The current separately exported hierarchical repair is required for final uncertainty.
- ARCHS4: a SHA-verified map replaces only `git_source(path)` immutable artifact lookup; byte-exact source modules are otherwise unchanged. No incomplete Git repository or fabricated history is distributed.
- T-cell1B/1D retain real original-history and CUDA gates. They require the recorded original Git worktree/history and frozen inputs; they are not standalone ZIP runs. No provenance gate is bypassed.

## Capability and runtime index

| Bundle | Actual class | Files | Input/runtime contract |
|---|---|---:|---|
| [experimental_compression](bundles/experimental_compression/README.md) | FULL_ANALYSIS | 34 | PREPARED_INPUTS_REQUIRED; full 50-target grid and 10000 hierarchical draws recomputed on 2026-09-08 |
| [low_rank_completion](bundles/low_rank_completion/README.md) | FULL_ANALYSIS | 22 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [m1_secondary](bundles/m1_secondary/README.md) | FULL_ANALYSIS | 11 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [context_anchoring](bundles/context_anchoring/README.md) | FIGURE_FROM_FROZEN_PREDICTIONS | 10 | CORRECTED_FROZEN_UTILITIES_REQUIRED; paired contrasts recomputed on 2026-09-08 |
| [mechanism_aligned](bundles/mechanism_aligned/README.md) | FULL_ANALYSIS | 29 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [tahoe_operator_anatomy](bundles/tahoe_operator_anatomy/README.md) | FULL_ANALYSIS | 39 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [tahoe_replicate_core](bundles/tahoe_replicate_core/README.md) | FULL_ANALYSIS | 41 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [held_context_scaling](bundles/held_context_scaling/README.md) | FULL_ANALYSIS | 13 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [five_pathway_scaling](bundles/five_pathway_scaling/README.md) | FULL_ANALYSIS | 10 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [ed7_hierarchical_repair](bundles/ed7_hierarchical_repair/README.md) | FULL_ANALYSIS | 4 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [tcell_truth](bundles/tcell_truth/README.md) | FULL_ANALYSIS | 32 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [tcell_lodo](bundles/tcell_lodo/README.md) | FULL_ANALYSIS | 31 | ORIGINAL_HISTORY_AND_CUDA_REQUIRED; not runnable from code-only ZIP |
| [tcell_decomposition](bundles/tcell_decomposition/README.md) | FIGURE_FROM_FROZEN_PREDICTIONS | 32 | ORIGINAL_HISTORY_AND_CUDA_REQUIRED; not runnable from code-only ZIP |
| [state_official_summary](bundles/state_official_summary/README.md) | OFFICIAL_ARTIFACT_SUMMARY | 15 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [state_independent_stability](bundles/state_independent_stability/README.md) | OFFICIAL_ARTIFACT_SUMMARY | 10 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [lea_historical_analysis](bundles/lea_historical_analysis/README.md) | FULL_ANALYSIS | 20 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [lea_archs4](bundles/lea_archs4/README.md) | FULL_ANALYSIS | 14 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [lea_full_span](bundles/lea_full_span/README.md) | FULL_ANALYSIS | 7 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [resolution_pathway_frozen](bundles/resolution_pathway_frozen/README.md) | FIGURE_FROM_FROZEN_PREDICTIONS | 23 | CORRECTED_CONTRAST_INPUT_INCLUDED; matched gene/pathway contrasts recomputed on 2026-09-08 |
| [pathway_fidelity](bundles/pathway_fidelity/README.md) | FIGURE_FROM_FROZEN_PREDICTIONS | 16 | PREPARED_INPUTS_REQUIRED; all-but-one predictions unchanged; replay protocol corrected |
| [pathway_composition](bundles/pathway_composition/README.md) | FIGURE_FROM_FROZEN_PREDICTIONS | 19 | PREPARED_INPUTS_REQUIRED; all-but-one predictions unchanged; replay protocol corrected |
| [crc_pdo_application](bundles/crc_pdo_application/README.md) | FULL_ANALYSIS | 15 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [crc_pdo_conditioning](bundles/crc_pdo_conditioning/README.md) | FULL_ANALYSIS | 13 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [crc_pdo_pca](bundles/crc_pdo_pca/README.md) | FULL_ANALYSIS | 12 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [crc_pdo_dense](bundles/crc_pdo_dense/README.md) | FULL_ANALYSIS | 13 | PREPARED_INPUTS_REQUIRED; scientific run not performed |
| [lea_semisynthetic](bundles/lea_semisynthetic/README.md) | FULL_ANALYSIS | 15 | PREPARED_INPUTS_REQUIRED; scientific run not performed |

## Test scope and environment

The latest local smoke suite reports 43/43 successful checks. It includes syntax/import/parser checks and tiny synthetic invariance/Gram/guard tests only. Those original smoke checks did not rerun scientific analyses. The separate 2026-09-08 support-grid rerun and linked corrections are documented in SUPPORT_BUDGET_CORRECTION.md.

SMOKE_ENVIRONMENT.json records the tested Python/package versions. Frozen pyproject files preserve historical dependency ranges; they are not complete lockfiles and do not prove cross-platform numerical equivalence. Additional optional libraries, CUDA for gated T-cell runs and third-party resource licenses remain reader prerequisites.

The separate formatted XLSX generator requires Node.js and `@oai/artifact-tool`; availability of that author-tested runtime outside this environment is not guaranteed. CSV table sources are independently usable without that tool. None of these utilities reconstructs author-edited SVG artwork byte-for-byte.
