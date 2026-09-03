# CGC Context Anchoring Efficiency Audit

Status: **FROZEN BEFORE ANCHORING OUTCOME INSPECTION**

Analysis identity: `POST_HOC_SECONDARY_FROZEN_PREDICTION_ANALYSIS`

## Scientific question

Using only the already-frozen strict prospective Entrywise Experimental Compression predictions, this analysis asks how much additional recovery of an unseen intervention is obtained from each additional measured response in the held-out target context. It tests whether the context axis admits a compact empirical susceptibility anchor or instead requires gradual, intervention-specific calibration.

This is a reporting-only secondary analysis. It does not train, refit, retune, or select any estimator, support set, sentinel set, context subset, seed, target, or outcome-dependent grid point.

## Frozen authority and source contract

- Authority branch: `codex/cgc_entrywise_experimental_compression`.
- Frozen scientific result commit: `dad1ca9dc86a6e0104976026810a246c20feba52`.
- Provenance-only authority descendant used to create this worktree: `a893cc338c11f402a9ef9966bd376f2aca92cb88`.
- Primary estimator: `M2_AFFINE_RIDGE`.
- Target: frozen context-specific excess response recovery `g_context_specific`.
- Original hierarchy: 50 target contexts, 93 interventions nested within each sampled context, with Plate 6/Plate 14 pairing already preserved in the per-episode frozen utility values.
- Bootstrap draws: the exact 10,000 frozen paired hierarchical draws created with seed `202608225`; no prediction or utility is recomputed.

The source audit must reproduce, to absolute tolerance `1e-12`:

- `g(40,4) = 0.07718711664021238`;
- `g(49,92) = 0.11068053088808294`.

Failure returns `CONTEXT_ANCHORING_SOURCE_MISMATCH` and stops before inference.

## Frozen support levels

Reference-context support is fixed at:

`m = [4, 16, 40]`

These represent low, intermediate, and high reference support. They may not be replaced by other levels after outcome inspection.

Primary target-context anchoring is fixed at:

`k = [0, 1, 2, 4, 8]`

The full frozen descriptive saturation grid is:

`k = [0, 1, 2, 4, 8, 16, 32, 64, 80, 92]`

## Frozen estimands

For each fixed `m` and `k in [1,2,4,8]`, the primary paired contrast is

`delta_g_anchor(m,k) = g(m,k) - g(m,0)`.

The early anchoring summaries are

`A8(m) = g(m,8) - g(m,0)`

and

`E8(m) = A8(m) / 8`.

The four marginal blocks are `0->1`, `1->2`, `2->4`, and `4->8`. For a block `(k1,k2)`,

`delta_block = g(m,k2) - g(m,k1)`

and

`eta = delta_block / (k2-k1)`.

These quantities are empirical anchoring gains, not information bits.

## Frozen paired inference

All contrasts use the same frozen target episodes and the same bootstrap draw in both terms. Pointwise paired 95% intervals use the 2.5th and 97.5th percentiles of the paired contrast draws.

The primary multiplicity family contains exactly the 12 `delta_g_anchor` contrasts:

`3 prespecified m levels x 4 prespecified k contrasts`.

A two-sided studentized maximum-absolute-deviation statistic is computed across all 12 contrasts in each bootstrap draw. Its 95th percentile defines one simultaneous 95% family-wise confidence interval for every contrast. The family, standardization, quantile, and draw set are fixed before outcome inspection. Pointwise p-values are not used for adjudication.

Marginal-block intervals are paired descriptive intervals and are not substituted into the 12-member primary family.

## Frozen structural adjudication

The final report returns exactly one verdict.

`COMPACT_CONTEXT_ANCHOR_SUPPORTED` requires all of the following:

1. at least two of the three prespecified `m` levels have a simultaneous lower bound above zero at `k=1` or `k=2`;
2. at least two levels recover at least 75% of their observed `A8(m)` by `k=2`, operationalizing a few-shot jump followed by early saturation;
3. no clear sign contradiction occurs across the three levels at the qualifying early contrast.

`CONTEXT_ANCHORING_GRADUAL_NOT_COMPACT` applies if the compact rule fails but at least two of the three `m` levels have a simultaneous lower bound above zero at `k=4` or `k=8`. This denotes reproducible empirical calibration without a compact one- or two-anchor code.

`SAME_CONTEXT_ANCHORING_WEAK` applies if neither positive rule is met. The report additionally states whether every descriptive `g(m,92)` remains below the original frozen 25% recovery target; that diagnostic bounds the meaning of weak early anchoring but does not change the verdict rule.

These rules distinguish lack of evidence for a compact anchor from failure of the broader biological context-operator hypothesis.

## Figure and reporting contract

One Extended Data/Supplementary four-panel figure is permitted:

- **a**, early `g(m,k)` curves for `k <= 8`;
- **b**, paired `delta_g_anchor` estimates with simultaneous 95% family-wise intervals;
- **c**, marginal per-anchor efficiency for the four fixed blocks;
- **d**, full descriptive saturation curves through `k=92`.

The figure uses Nature Biotechnology-style restrained typography and line weights, lowercase panel letters, an unequal hierarchical layout that gives the primary early-anchoring panels more space, editable SVG, high-resolution PNG, and reconciled source data. It is not automatically promoted to a main figure.

## Stop rule

After this audit is frozen, stop. Do not design a new sentinel or context-anchor model; optimize sentinel identities; search context subsets; alter `m`, `k`, estimator, target, or bootstrap; or add experiments to obtain a preferred IGC-CGC symmetry.
