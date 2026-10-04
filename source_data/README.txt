Source Data

The source tables support five main figures, three Extended Data figures and four supplementary figures. CURRENT_PANEL_MAP.csv maps tables to the corresponding panels.

Figure 5 includes the clone-1978 fate example, the 21-clone paired-error analysis, the normalized error bound, the information comparison and the reliability-criterion comparison. Extended Data Figure 3 provides culture-family estimates, CellTag calibration and paired contrasts.

Negative cross-library estimates are retained. LARRY intervals resample cells conditional on observed cultures and libraries. The paired-error floor is normalized to reproducible endpoint variation; its complementary fraction includes variation in branch midpoints whose predictability from the early record is not assumed.

CellTag set sizes and held-out coverage are reported together. Nominal 80% and 90% marginal and fate-conditional criteria use saved predictions and calibration cutoffs. ATAC-only and shuffled-correspondence controls are included.

Clone-level contrasts and sensitivity results are in Extended_Data_3_RNA_contrasts.json and Extended_Data_3_robustness.json. Cohort-level intervals use the named LARRY tables. Analysis code is in code/analysis and code/scientific; processed inputs and conditional draws are in prepared. PROGENy_weights.csv contains pathway weights, and EC_support_grid.csv contains the experimental-support grid.
