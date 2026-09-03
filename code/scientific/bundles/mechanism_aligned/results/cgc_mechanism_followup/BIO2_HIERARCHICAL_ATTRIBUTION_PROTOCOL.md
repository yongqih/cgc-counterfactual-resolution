# CGC-BIO-2 Hierarchical Attribution Protocol — FROZEN

Frozen before any CGC-BIO-2 outcome calculation on 2026-08-22. Base commit:
`6d3b3203504eb7f18917c949cfd94536d6f5648d`.

## Scientific estimand

Locate the highest biological resolution at which the already-frozen Tahoe q49
LOCO-affine unresolved residual is organized. This is an attribution-boundary
audit, not an attempt to preserve or overturn the historical BIO-1 verdict.

## Immutable inputs

- Plate6 and Plate14 remain separate throughout.
- 50 frozen contexts, 93 exact drug-dose interventions, 25,695 G_PRIMARY genes.
- Primary response is `residual` in the hash-verified BIO-1 gene cache.
- Confirmatory representations are the separately scaled frozen PROGENy (14)
  and CollecTRI (1,153) spaces. Reactome is secondary because it also defines
  420 of the 424 genotype-pathway relations.
- No q49 operator, response, context, intervention, gene, or program is refit.

## Frozen pharmacologic hierarchy

The ordered ladder is:

1. `BROAD_ACTION`: shared non-unclear BIO-1 `moa-broad` label.
2. `MOA`: shared non-unclear BIO-1 `moa-fine` label.
3. `TARGET_FAMILY`: at least one shared eligible HGNC gene-group ID, excluding
   pairs sharing an exact target.
4. `EXACT_TARGET`: identical non-empty canonical molecular target sets, matching
   the historical BIO-1 definition.
5. `GENOTYPE_DRUG`: correct frozen annotated alteration-drug relationship beyond
   the best valid same-drug/pharmacologic-neighborhood relation.

The indicators are a partial-order, not assumed perfectly nested. In particular,
GNRHR agonists and antagonists share an exact target set but not action direction.

## Fingerprints and plate replication

For intervention p and plate b, `F[b,p]` concatenates all 50 context residual
vectors in frozen context order. Similarity is cosine. Compute and retain:

- Plate6-derived `cos(F6_i, F14_j)`;
- Plate14-derived `cos(F14_i, F6_j)`;
- the arithmetic mean of the two directional values as pooled confirmation.

The same construction is applied independently in gene, PROGENy, and CollecTRI
spaces. No representation is concatenated or outcome-weighted.

## Conditional hierarchy

The primary incremental statistic is the coefficient for each ladder indicator
in one predeclared pair model containing all lower-resolution indicators and the
frozen nuisance terms. Pairwise standard errors are never used. Inference uses a
10,000-draw intervention-label quadratic-assignment permutation and a 10,000-draw
intervention-cluster bootstrap (`seed=7201`). The full annotation row moves as a
unit under permutation so target/MOA dependence is preserved.

Matched-control contrasts are a co-primary interpretability analysis and use the
fixed rules in `BIO2_MATCHING_RULES.json`. A level may be declared robust only if
the conditional and matched effects agree in direction.

## Genotype hierarchy

All 424 BIO-1 `formal_eligible` relations are retained. They are labeled without
reinterpretation as 420 pathway-linked and four direct target-linked relations.
The primary G3-G2 statistic compares the correct relation with wrong alteration
relations for the same drug when an exchange block exists (420/424 relations).
Singleton-drug relations use the frozen nearest block: exact target set, then fine
MOA, then broad action. The four direct relations are descriptive sensitivity
only and cannot support an independent precision-genomic claim.

Permutation is at the drug/alteration block, never at the row level. Mutant and
wild-type counts, lineage composition, relation multiplicity, response magnitude,
and residual energy remain matched as specified in the rules file.

## Robustness and decision gate

Run leave-one-drug, leave-one-MOA, leave-one-target-family, and leave-one-lineage
analyses without changing eligibility. Report median, minimum, maximum, and sign
consistency. A ladder level is `ROBUST` only if all are true:

- pooled conditional effect is positive with 95% cluster-bootstrap lower bound > 0;
- blocked/QAP permutation p < 0.05;
- Plate6→Plate14 and Plate14→Plate6 effects have the same positive direction;
- leave-group-out sign consistency is at least 0.80 for every estimable grouping;
- at least five independent annotation clusters contribute.

Three or four clusters are `LIMITED_POWER`; fewer than three are
`NOT_ESTIMABLE`. Exact target is known a priori to have only three clusters and
five positive pairs, so it cannot be promoted to robust attribution even if its
point estimate is large. The final product is `HIGHEST_ROBUST_ATTRIBUTION_LEVEL`,
not a global PASS/FAIL.

## Prohibited changes

No outcome-selected pathway, family, nuisance, caliper, representation, null,
threshold, permutation unit, or subgroup may be added after this freeze.
