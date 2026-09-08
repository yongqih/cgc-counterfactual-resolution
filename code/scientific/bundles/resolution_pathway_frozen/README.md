# Same frozen predictions, gene-versus-predefined-pathway readout and corrected inference

Capability: `FIGURE_FROM_FROZEN_PREDICTIONS`. Source: `46b25eec4a97e3001cc43262980af8f69c533478`.

Runtime: **CORRECTED_CONTRAST_INPUT_INCLUDED; matplotlib/Arial runtime required**.

The source commit above identifies the historical base. On 2026-09-08, episode-reference-only hyperparameter selection and its downstream replay were corrected. SOURCE_MANIFEST.json distinguishes corrected bytes from the original Git blobs and retains their pre-correction hashes. Historical result tables must not be substituted for corrected results. The public correction and reproduction guide is ../../SUPPORT_BUDGET_CORRECTION.md.

## Safe inspection

```text
python code/scientific/launch.py resolution_pathway_frozen --inspect-only
```

The inspection command is not analysis reproduction. To execute a scientific stage, provide its exact prepared inputs and deliberately add `--execute`. The 2026-09-08 corrected runs and unchanged analyses are distinguished in ../../SUPPORT_BUDGET_CORRECTION.md.

## Entry points

- `python code/scientific/launch.py resolution_pathway_frozen --execute -- --out-dir <new-output-dir>`

For allowlisted argument-parsing scripts, replace `--execute` and arguments with `--cli-help`. Scripts without a parser must never be probed with raw `--help`: several historical scripts would start their analysis. The launcher rejects unavailable help safely.

## Input contract

- `results/cgc_resolution_poc_v2/RESOLUTION2_VALID_COMPARISONS_CORRECTED.csv` — included corrected six-row gene/pathway result table; this is the current renderer input.

Prepared paths are relative to the workspace or explicit source root used by the original CLI. The registry is not a claim that those data are embedded. Large inputs were neither copied nor downloaded. Use the source manifests and DATA_AVAILABILITY_FINAL.md to resolve provenance before running.

## Boundaries

- Frozen source export is not evidence that scientific execution has been rerun.
- Third-party arrays are not included. Run only after arranging original input paths and verifying frozen checksums.
- pyproject.toml is a dependency range declaration, not a fully pinned environment lock; matplotlib and some historical optional runtime dependencies need explicit installation.
- Historical random-projection arm is INVALID and excluded from manuscript inference. Source inclusion is archival, not endorsement or permission to execute that arm.
- Valid pathway-minus-gene comparison is 0.27281393788398667 versus 0.11068053088808294; the claim authority is the corrected frozen interval, not random-projection results.
- Default source/truth roots are author-machine Windows paths; override them explicitly. No automatic replay or scientific execution during packaging.
- claim_boundary: The current renderer prefers RESOLUTION2_VALID_COMPARISONS_CORRECTED.csv: corrected support-budget predictions and the two retained pathway-minus-gene comparisons under the original four-term max-T calibration. Historical random terms are retained solely in the multiplicity maximum, not as biological control evidence. Historical tables are preserved for provenance, not current manuscript inference. Manual R2 layout is not reproduced.
- active_execution_guard: Original all-stages scripts cannot be invoked through release launcher; only current bounded entrypoint is allowed.
