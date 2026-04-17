# Session memory
Updated: 2026-04-17 11:30

## Current task
Regression tests for stage 02 scripts — in progress after archiving dead script.

## Context / why
Scripts had all logic at module level; adding main() functions so regression tests can call them with lsae_1209 golden inputs and compare against golden files.

## Where we are
- Script 03 (make_covariate_means) was found to be dead — nothing downstream reads covariate_means.nc — and archived to archive/02_data_prep/
- Scripts renumbered: old 04–10 → new 03–09; tests renumbered to match
- constants.py and versioning.py cleaned of COV_MEANS / _A03_COV_MEANS references
- test_03_make_covariate_means.py archived (was written but never passed)
- New numbering: 03=rake_aa, 04=rake_as, 05=malaria_modeling_df, 06=dengue_modeling_df, 07=forecasted_non_draw, 08=forecasted_malaria_parallel, 09=forecasted_dengue_parallel

## Next steps (ordered)
1. Commit current changes (archive + renumber + notebook)
2. Write test_03_rake_aa_A2_to_GBD.py (was test_04)
3. Write test_04_rake_as_A2_to_GBD.py
4. Continue through 05, 06, 07
5. After all tests pass: run scripts with lsae_1285 to generate versioned outputs
6. Scripts 08–09 deferred until stage 04 restructuring

## Resume prompt
Refactor branch feature/refactor-shared-lib. Stage 02 data prep regression tests in progress.
Script 03 (make_covariate_means) was archived as dead code — nothing reads its output.
Scripts renumbered 04–10 → 03–09. Next: write test_03_rake_aa_A2_to_GBD.py using the same
pattern as the archived test_03 (importlib loads main(), fixture calls with lsae_1209 golden inputs,
compare output against golden in /mnt/team/idd/pub/forecast-mbp/02-processed_data/raked_aa/).
