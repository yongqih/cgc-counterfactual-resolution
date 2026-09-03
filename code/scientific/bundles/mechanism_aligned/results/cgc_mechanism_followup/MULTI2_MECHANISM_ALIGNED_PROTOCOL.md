# CGC-MULTI-2 Mechanism-Aligned Susceptibility Protocol — FROZEN

Frozen before any MULTI-2 outcome calculation on 2026-08-22. Base commit:
`6d3b3203504eb7f18917c949cfd94536d6f5648d`.

## Question

Does an outcome-independent intervention-aligned baseline state `Z[c,p]` recover
more of the frozen q49 context-specific residual than the same modality supplied
as generic context PCs `Z[c]`? Dependency is an explicitly separate functional
perturbational oracle.

## Immutable response and evaluation

- Primary target: BIO-1 frozen gene residual, Plate6/Plate14 × 50 × 93 × 25,695.
- Secondary target: frozen full response (`gamma`).
- Strict outer leave-one-context-out (LOCO), plate directions kept separate.
- All preprocessing, response PCA, generic-modality PCA, ridge selection, and
  intervention memory are fitted using outer-training contexts only.
- Response ranks are `{8,16,32,64}` and are selected by fixed five-fold inner
  intervention CV (`intervention_axis % 5`). Ridge grid is
  `{1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100}`.
- Primary utility is paired cross-plate `g`; 10,000 bootstrap draws resample only
  the frozen OOF context×intervention numerator/denominator table.

## Honest complete cases

Historical MULTI-1 `complete_rows` meant any finite feature and then median-filled
missing columns. MULTI-2 does not call that complete-case.

For each modality, first restrict to the predeclared context set with an official
profile. Select feature columns observed in every context in that set before any
response is loaded. An aligned `(context, intervention)` value is eligible only
when the intervention has a frozen target/pathway mapping and at least one mapped,
fully observed molecular feature. Missing annotation/measurement is never encoded
as zero. Each M0–M4 comparison uses the exact same contexts and interventions.

Ordinary modalities use matched per-modality panels. Wiring+RPPA+Sanger uses a
predeclared 40-context intersection for the joint M5 comparison. Dependency uses
its separate 39-context oracle panel and never shrinks ordinary-modality panels.
The all-six 30-context panel is sensitivity only.

## Frozen model comparisons

- M0: RNA only.
- M1: generic raw modality only.
- M2: mechanism-aligned modality only.
- M3: RNA + generic modality.
- M4: RNA + mechanism-aligned modality.
- M5: RNA + aligned wiring + aligned activity.
- M_ORACLE: RNA + aligned dependency, labeled
  `FUNCTIONAL_PERTURBATIONAL_ORACLE`.

Both additive multi-output Ridge and compact Bilinear Ridge are run. The bilinear
design exposes every generic or aligned feature to the fixed training-only
intervention embedding; it does not repeat MULTI-1's first-eight-RNA-PC truncation.
No estimator, architecture, rank, feature, or regularization value is added after
outcomes.

## Primary mechanism-aligned features

Multi-target values use the arithmetic mean of available frozen target features.
The max aggregation is a predeclared sensitivity. Mutation is the mean binary
alteration burden (hotspot/damaging); CNV remains continuous. Frozen Reactome
membership supplies pathway sets. Pathway-aligned RPPA is the powered activity
primary (47 pre-audit mapped interventions); exact-target RPPA/protein/phospho are
sparse secondary axes (7/27/5 interventions respectively).

The low-dimensional feature map is fixed in `MULTI2_FEATURE_MAP.json`. Generic
comparators use training-only PCs from the exact same raw modality and context
panel, ensuring the alignment comparison changes correspondence rather than data.

## Nulls

Run 10,000 frozen-table null draws (`seed=8201`):

- drug-target mapping permutation, blocked by target-set size, MOA frequency,
  annotation coverage, and response-magnitude quintile;
- pathway mapping permutation, blocked by pathway-set size and coverage;
- context-modality permutation within frozen lineage blocks;
- random gene-set controls matched on set size and molecular measurement coverage.

Null mappings are generated before fitting and never selected by their outcome.

## Positive controls and power

For every modality, inject true incremental `delta_g` of 0.02, 0.05, 0.10, and
0.20 using the exact panel, splits, feature map, mechanism mapping, and covariance.
Use 200 replicates per level plus a zero-signal FPR condition. Batched sufficient
statistics must replace per-replicate full-gene fitting. Report MDE80, power,
false-positive rate, bias, and CI coverage. A real null supports biological absence
only at effect sizes at or above that modality's calibrated MDE80.

## Heterogeneity

Freeze OOF gains per intervention and aggregate by frozen target family and MOA
using inverse-variance random-effects summaries. No subgroup is selected from its
gain. Coherent subgroup gains are reported even when the global mean is null.

## Prohibited changes

No outcome-driven target/pathway mapping, feature selection, imputation, context
or intervention filtering, rank rescue, hyperparameter rescue, estimator addition,
or GPU use while the frozen CMonge audit is running.

