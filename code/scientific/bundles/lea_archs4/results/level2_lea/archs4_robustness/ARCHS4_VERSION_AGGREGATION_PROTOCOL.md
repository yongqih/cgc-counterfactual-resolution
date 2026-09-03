# ARCHS4 matched-340 version aggregation and representation protocol

Frozen before ARCHS4 or matched post-SVA model outcomes were calculated.  
Coverage authority: `35cf07d8f3d90450b365893b56aa127dca7c4774`  
Branch: `codex/lea_archs4_robustness`  
Dataset object: ARCHS4 human gene v2.5, `human_gene_v2.5.h5`, remote ETag `350f7e2f1096e77b8b8da50c6a91c509-9130`.

## Frozen biological core

- Retain exactly the 340 title-resolved historical paired LCLs.
- Permanently exclude `LineNA` and `Line223`; no heuristic identity rescue is allowed.
- Apply the exact historical five-fold LCL assignment from `LEVEL2_OOF_FROZEN.npz` after deleting the two excluded rows. No reshuffling or rebalancing.
- Every retained LCL contributes one ETOH vector and one DEX vector downstream and therefore one unit of model weight.

## Frozen per-GSM transformation

For each of the 1,026 ARCHS4-v2.5-present candidate GSMs assigned confidently by the frozen GEO-title rules:

1. Read the Kallisto-derived ARCHS4 integer pseudocount column over all 67,186 ARCHS4 genes.
2. Define the library size as the sum over all 67,186 genes. A non-finite or zero library fails the representation gate; it is not imputed or removed post hoc.
3. Compute counts per million with the fixed constant 1,000,000.
4. Apply the natural-log variance-stabilizing transform `log1p(CPM)` independently to each sample.
5. Select the frozen one-to-one matched GeneID axis only after library-size normalization, so the denominator is not changed by matching coverage.

No cohort statistic, DEX outcome, held-fold statistic, or model outcome enters this transformation. No alternative normalization is permitted after outcomes are seen.

## Frozen raw-version aggregation

Within every LCL-by-treatment group, take the arithmetic mean of all confidently assigned ARCHS4-present GSM log1p-CPM vectors with equal weight:

`X_line,treatment = mean_GSM(log1p(1e6 * counts / full_library_size))`.

Do not select a deepest, best-correlated, v1, `.x`, `.y`, or otherwise preferred GSM. Do not sum pseudocounts across versions. Different version multiplicities do not change the LCL's downstream model weight.

## Frozen gene axis

- Historical axis authority: first-appearance order of the 10,120 unique Ensembl GeneIDs in the 10,157-row Lea matrix.
- ARCHS4 key: exact version-free `meta/genes/ensembl_gene` string.
- Retain only exact one-to-one identifiers present once in ARCHS4 and at least once historically.
- ARCHS4 duplicate GeneIDs fail closed. Missing or ambiguous identifiers are excluded before outcomes.
- For the matched post-SVA control, multiple historical rows sharing one GeneID are collapsed by an arithmetic mean across those rows. This outcome-independent rule yields one post-SVA feature for the same GeneID represented once in ARCHS4.
- Missing genes are not zero-filled in the primary comparison; both representations use the same maximum safe exact intersection in the same order.

Unexpectedly poor exact coverage triggers review before modeling. The coverage gate is evaluated only from identifiers, never expression outcomes.

## Frozen target and models

- Predictor: the LCL's ETOH vector.
- Response: DEX minus ETOH.
- Residual target in each historical outer fold: response minus the outer-training LCL mean response.
- Full-gene models: exact historical Ridge, PCA+Ridge, MLP, exact RBF kernel Ridge, histogram gradient boosting, and deep residual MLP implementations from commit `c3d074412ecc1be0958141726db16a1ed0036a04`; their historical inner-CV grids, seeds, and training rules remain unchanged.
- Program analysis: fold-local 16-component response PCA fit on outer-training responses only; project the exact RBF kernel-Ridge OOF prediction and truth, then report pooled PC1-16, PC1-8, and PC1-4 residual R2 exactly as the historical response-program evaluator does.
- Run the same battery separately for matched post-SVA and ARCHS4. No model sees or pools the other representation.

## Leakage rule

Held-out DEX values may affect only held-out truth and downstream scoring. They must not affect normalization, version aggregation, gene matching, fold identity, training-mean response, PCA bases, hyperparameter selection, fitted parameters, or predictions. The implementation must mutate held-out DEX and demonstrate identical fit-side hashes and predictions before formal interpretation.

## Extraction and cache boundary

Use HTTP byte ranges against the v2.5 HDF5 and store only selected GSM/matched-gene log1p-CPM values plus 340-by-treatment aggregates. Do not download the full HDF5, SRA, FASTQ, or BAM. A resumable range cache is allowed only for selected HDF5 blocks. Report transferred bytes, cache footprint, runtime, peak RAM, and GPU use.

## Replicate boundary

Raw `.x/.y` correspondence is not recoverable for all ambiguous version groups. The primary all-version equal-mean aggregation is not a replicate reconstruction. Unless exact pairing emerges independently of outcomes, report `ARCHS4_REPLICATE_TRUTH_RECONSTRUCTION_NOT_EXECUTED_DUE_TO_RAW_VERSION_IDENTITY_AMBIGUITY`.

## Frozen numerical adjudication

These operational rules are frozen before either matched representation is modeled. They translate the task's qualitative terms into an outcome-independent decision and do not create a new verdict class.

For each representation define

`resolution_gap = R2_PC1_4 - R2_best_full_gene`.

Treat the coarse-to-fine ordering as present when `PC1-4 >= PC1-8 - 0.01`, `PC1-8 >= PC1-16 - 0.01`, and all three program values exceed the best full-gene value. The `0.01` tolerance prevents negligible sampling/numerical reversals among adjacent program resolutions from changing the scientific verdict.

- `ARCHS4_RESOLUTION_HIERARCHY_REPLICATED` requires: (i) matched post-SVA `resolution_gap >= 0.05`; (ii) ARCHS4 `resolution_gap >= 0.05`; (iii) the coarse-to-fine ordering is present in both representations; and (iv) the ARCHS4 gap is at least 50% of the matched post-SVA gap. Thus a clear hierarchy may change quantitatively but cannot be called replicated if most of its separation disappears.
- `ARCHS4_RESOLUTION_HIERARCHY_PARTIAL` requires: (i) matched post-SVA `resolution_gap >= 0.05`; (ii) ARCHS4 `resolution_gap >= 0.02`; (iii) ARCHS4 PC1-4 exceeds the best full-gene value; and (iv) at least two of the three ARCHS4 program values exceed the best full-gene value, while one or more full-replication requirements fail.
- `ARCHS4_RESOLUTION_HIERARCHY_NOT_REPLICATED` applies to every executable result not meeting the replicated or partial rules.
- `ARCHS4_ROBUSTNESS_NOT_EXECUTABLE` remains reserved for a genuine data-validity or technical failure before a clean matched comparison can be completed.

The hierarchy is adjudicated from pooled outer-fold predictions. Genes are not treated as biological replicates, and no post-outcome alternative threshold is permitted.
