<!-- DRAFT — generated 2026-05-06 17:30 without review -->
<!-- Claude notes:
  - We were mid-run on 05_build_malaria_past_inputs.py when Bobby bailed
  - The xarray/netCDF restructure is a new initiative not yet reflected in STATUS.md
  - "Next steps" from old STATUS.md are mostly superseded by this session's work
  - The old script 05 and 06 were replaced (git rm); new scripts created
  - Parking lot item "malaria suitability curve dimension" is now implemented in new script 05
  - Main blocker: wide_to_array bug (index vs columns in climate parquets)
-->

# Project status
Updated: 2026-05-06

## Goals
Infectious disease forecasting pipeline for malaria and dengue, projecting outcomes to 2100 under SSP climate scenarios + DAH funding scenarios. Currently executing:
1. **[IN PROGRESS]** Restructure pipeline storage from ~1,500 parquet files (~5TB) to dimension-aware xarray/netCDF files — past inputs first, then future inputs/outputs
2. Run reference scenario end-to-end with new data and new storage structure
3. Eventually: vaccination scenario, variable importance methods, full sensitivity framework

## Recent steps
- 2026-04-20: Added main() + regression tests for stage 04 Python compute scripts — all 5 tests pass
- 2026-04-20: Versioned 03-modeling_data/dengue_cfr_model/lsae_1209/ for R-generated dengue CFR artifacts
- 2026-05-06: Designed xarray/netCDF dimensional structure for past and future inputs/outputs
- 2026-05-06: Replaced old 05_malaria_modeling_dataframe.py and 06_dengue_modeling_dataframe.py with new 05_build_malaria_past_inputs.py and 06_build_dengue_past_inputs.py (netCDF output, xarray, all 14 suitability variants, 100 draws)
- 2026-05-06: Added `_A03_MAL_PAST_INPUTS` and `_A03_DEN_PAST_INPUTS` artifact roots to constants.py and versioning.py; added `lib/io/array_builders.py` with wide_to_array, scalar_to_array, read_shared_covariates, read_draw_climate

## Next steps
1. Fix `wide_to_array` in `lib/io/array_builders.py` — climate parquets have loc/year as MultiIndex not columns; fix: read `columns=DRAWS` only, then `reset_index()` before `set_index(['location_id','year_id'])`
2. Run `05_build_malaria_past_inputs.py` to completion (fix any remaining errors)
3. Run `06_build_dengue_past_inputs.py`
4. Write regression tests for both scripts
5. Design future input/output file structure (not yet started)

## Parking lot
- Delete legacy modules (parquet_functions, xarray_functions, helper_functions, hd5_functions) — after refactor fully verified
- combine_as_draws.py dead-end — revisit when draw combining becomes a bottleneck
- Stage 06 (upload): never verified against goldens — run compare_outputs.py before relying on it
- Stage 05 orchestrator and stage 06 scripts: `finalize_artifact` still deferred (listed in DEFERRED in test_versioning_completeness.py)
- Dengue revision (after malaria paper accepted)
- Feature: vaccination scenario dimension
- Feature: variable importance methods beyond "hold at 2022"
