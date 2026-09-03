# CGC-BIO-2 Frozen Null Definitions

All randomization uses NumPy PCG64 with root seed 7201 and 10,000 draws.

## Intervention hierarchy

- `QAP_ANNOTATION`: permute the complete intervention annotation rows relative
  to the fixed similarity and nuisance matrices within frozen target-count and
  dose-scale strata. This preserves label prevalences and dependencies among
  broad action, fine MOA, target family, and exact target.
- `MATCHED_UNRELATED`: for every positive pair, compare its similarity with the
  mean of up to five deterministic nearest eligible lower-resolution controls.
- `CLUSTER_BOOTSTRAP`: resample intervention identities, reconstruct all induced
  eligible pairs, and refit the frozen conditional model. Duplicate self-pairs
  are discarded.

Two-sided permutation p-values use `(1 + count(|null| >= |observed|))/(10001)`;
the preregistered positive-direction gate additionally reports the one-sided p.

## Genotype-drug hierarchy

- `G0`: alteration labels exchanged across matched prevalence/coverage blocks.
- `G1`: wrong alteration pairing within lineage and exact drug where possible.
- `G2`: wrong alteration pairing within exact drug; if impossible, use the first
  available frozen neighborhood in order exact target set → fine MOA → broad action.
- `G3`: the correct frozen annotated relation.

The primary contrast is G3-G2. Exchange units are drug clusters and alteration
genes; relation rows are never treated as independent. Direct target-linked
relations (`n=4`) are descriptive only.

## Plate and representation multiplicity

Gene-space pooled G3-G2 and the pharmacologic ladder are primary. PROGENy,
CollecTRI, and both plate directions are confirmation axes. No result-dependent
multiplicity scheme is introduced; all raw p-values and Benjamini-Hochberg q-values
within each predeclared family are reported.

