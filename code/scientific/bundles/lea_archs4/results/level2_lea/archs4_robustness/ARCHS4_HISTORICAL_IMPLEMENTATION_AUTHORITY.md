# Recorded Lea implementation provenance

The Lea response analysis uses implementation commit `c3d074412ecc1be0958141726db16a1ed0036a04`:

| role | source path | Git blob |
|---|---|---|
| metadata and target centering helpers | `src/igc_virtual_cell/level2_lea.py` | recovered directly from the provenance commit |
| Ridge, PCA+Ridge, pilot MLP and pooled residual R2 | `scripts/level2_lea_pilot.py` | recovered directly from the provenance commit |
| exact RBF kernel Ridge, boosting and deep residual MLP | `src/igc_virtual_cell/level2_full.py` | `34bb1c95b0f1408096358d7fcc6f6d061449363f` |
| full model orchestration and summary metric | `scripts/level2_lea_full_battery.py` | `1cd95b97dd8490ae9f0e4ac2b9ee246719ba27c7` |
| fold-local response-program projection and pooled PC-k R2 | `scripts/level2_lea_response_programs.py` | recovered directly from the provenance commit |
| full model grid | `configs/level2_lea_full_battery_frozen.json` | `7ce919ccd38fcd7d31c9a0957bd334c067c17a37` |

The runner loads the recorded Git objects and uses two matched data representations with the same fold assignments. It supports resumable execution, source tracing, matched-gene construction and reporting.
