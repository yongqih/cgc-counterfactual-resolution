# Shared-record prediction error: locked analysis plan

Locked before aggregate uncertainty, normalization and new sensitivity results
were computed. The 21 individual raw estimates had already been inspected;
this is not a prospective preregistration.

## Target and inclusion

Use the existing 21 clones with at least 10 day-6 records in each well after
the existing repeated-record screen. Include all 21 regardless of separation.
The same recorded day-2 clonal RNA and nominal pre-split protocol are available
to both branches. Predictions may differ between clones, but a predictor with
this shared record must give the same point prediction to a clone's two wells.
The target is the mean log1p published normalized expression profile of each
realized descendant population, across all 25,289 genes.

## Estimands

For each clone, L = dot(A1-A2, B1-B2)/4, using the first sorted published library
in each well as A and the remaining libraries pooled as B, exactly as before.
Raw aggregate: equal weight per clone and per well. Also report three family
means and an equal-family descriptive mean. Families are observed culture
strata, not assumed independent donors. Do not bootstrap clones as if they
were independent culture experiments, and do not introduce a population P value.

Normalize by V: the A/B cross-product variance of clone-by-well endpoint
profiles around their own culture-family centroid, with the same weights.
Family centering avoids making a large between-family baseline difference
the denominator. By algebra V = midpoint variation + L. Report both terms,
the uncentered-across-family alternative as a sensitivity, and L/V without
clipping. This is a scale comparison within the observed cohort, not a
population Bayes-risk fraction or a deployable predictor. If V is unstable
or nonpositive, do not promote its ratio to the main figure.

## Uncertainty and robustness

Use 3,000 conditional cell bootstrap draws within clone x well x published
library, keeping observed library weights fixed. These intervals quantify
sampled-cell uncertainty conditional on the observed cultures and libraries;
they do not capture new cultures or systematic library biases. Preserve
negative cross-product estimates and do not count pointwise positive CIs as
multiple-testing-corrected clone discoveries.

Keep the fixed 21-clone cohort for the stricter barcode sensitivity. Report
lost cells and group sizes, and stop that comparison if any A/B group empties.
For a reference sensitivity, subtract same-library neutrophil mean profiles
from clones outside the entire 21-clone cohort. These reference cells are
disjoint from all targets. Resample reference cells once per library per draw
and reuse those draws for every target clone, propagating shared-reference
uncertainty. The reference is not assumed biologically invariant.

Also subtract the family-average well contrast as an intentionally aggressive
diagnostic. This removes shared biology as well as shared technical shifts;
it is not a correction to an assumed true state. Report the resulting L and V.
Report clone-1978 exclusion and all leave-one-clone-out point estimates.

## CellTag companion

Reuse saved held-out predictions and calibration cutoffs at nominal 80% and
90%; do not fit models. Verify split identities, probabilities, sets and
published summaries. Report mean set size together with empirical coverage
for prior, RNA, ATAC, matched joint and shuffled joint conditions, for both
existing model families. Use existing paired contrasts where available;
new prior contrasts use clone-averaged repeated predictions and paired
conditional clone bootstrap. More specific means fewer admissible fates.
No individual-conditional coverage or RNA/ATAC synergy claim.

## Stop rule

After these analyses, revise the Figure 5 evidence hierarchy and associated
text according to results. No new data collection, model sweep, or Tahoe
frontier promotion is part of this analysis.
