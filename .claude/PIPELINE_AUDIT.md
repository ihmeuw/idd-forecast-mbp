# Pipeline Audit
Generated: 2026-03-26

## Overview

Six pipeline stages. Stage 03 is R-only; stages 01, 02, 04, 05, 06 are Python. Each stage has a mix of core computation scripts and Jobmon orchestration scripts that parallelize across draws/scenarios on SLURM.

---

## Stage 01: `01_map_to_admin_2/` — Pixel-to-Admin Aggregation

**Purpose:** Convert raw GeoTIFF climate data to netCDF, then aggregate to admin-2 and higher levels using population-weighted averaging.

### Files

#### `01_prep_maps.py`
- **What it does:** Converts GeoTIFF files to netCDF format. Supports single-year synoptic variables and multi-year temporal data with optional global extent padding.
- **Inputs:** GeoTIFF files from COVARIATE_DICT paths; `COVARIATE_DICT.yaml`
- **Outputs:** netCDF files → `MODEL_ROOT/02-processed_data/cc_insensitive/`
- **Functions:** `geotiff_to_netcdf()`, `create_multiyear_netcdf()`, `batch_process_all_covariates()`, `set_file_permissions()`
- **Project imports:** `constants` (rfc), `yaml_functions`

#### `pixel_main.py`
- **What it does:** Maps pixel-level climate data to admin-2 regions using population-weighted aggregation.
- **Inputs:** CLI: `--covariate`, `--hiearchy` [sic], `--block_key`; netCDF climate files; GeoTIFF population; admin shapes parquet
- **Outputs:** Parquet → `MODEL_ROOT/02-processed_data/GBD2023/{hierarchy}/{covariate_name}/{block_key}/000.parquet`
- **Functions:** `get_bbox()`, `load_raking_shapes()`, `build_bounds_map()`, `build_location_masks()`, `pixel_main()`
- **Project imports:** `constants` (rfc), `yaml_functions`

#### `pixel_hierarchy.py`
- **What it does:** Rolls up pixel-level data from admin-2 to higher hierarchy levels.
- **Inputs:** CLI: `--covariate`, `--hierarchy`; block-level parquets; hierarchy parquet
- **Outputs:** Parquet → `MODEL_ROOT/02-processed_data/GBD2023/{subset_hierarchy}/{summary_covariate}_{scenario}.parquet`
- **Functions:** `aggregate_climate_to_hierarchy()`, `load_subset_hierarchy()`, `post_process()`, `hierarchy_main()`
- **Project imports:** `constants` (rfc), `helper_functions`

#### `pixel_urban_main.py`
- **What it does:** Classifies pixels as urban/rural by population density threshold; aggregates to admin-2.
- **Inputs:** CLI: `--threshold`, `--hiearchy`, `--block_key`; 100m and 1km population GeoTIFFs; admin shapes
- **Outputs:** Parquet → `MODEL_ROOT/02-processed_data/{hiearchy}/urban_threshold_{threshold}_simple/{block_key}/000.parquet`
- **Functions:** `get_bbox()`, `load_raking_shapes()`, `build_bounds_map()`, `build_location_masks()`, `pixel_main()` — **NOTE: same names as pixel_main.py, separate implementations**
- **Project imports:** `constants` (rfc), `yaml_functions`

#### `pixel_urban_hierarchy.py`
- **What it does:** Rolls up urban classification from admin-2 to higher hierarchy levels.
- **Inputs:** CLI: `--threshold`, `--hierarchy`; urban pixel parquets; hierarchy parquet
- **Outputs:** Parquet → `MODEL_ROOT/02-processed_data/{subset_hierarchy}/urban_threshold_{threshold}_simple_mean.parquet`
- **Functions:** `aggregate_climate_to_hierarchy()`, `load_subset_hierarchy()`, `post_process()`, `hierarchy_main()` — **NOTE: same names as pixel_hierarchy.py, separate implementations**
- **Project imports:** `constants` (rfc), `helper_functions`

#### `run_suitability_pipeline.py`
- **What it does:** Wraps the external `climate_data` repo pipeline to run custom temperature-suitability curves for malaria.
- **Inputs:** `malaria_temp_suitabilities_df.parquet`; external climate-data repo data
- **Outputs:** Suitability curve parquets; annual raster results; draw mappings; aggregated data in `climate_data` repo paths
- **Functions:** `run_custom_suitability_curves()`, `_make_custom_suitability_mapper()`, `_draws_main_with_mapping()`
- **Project imports:** `constants` (rfc); heavy use of `climate_data.*` (external repo)

#### Orchestrators (Jobmon/SLURM)
- `02_pixel_main_parallel.py` — Distributes `pixel_main.py` across covariate × hierarchy × block
- `03_pixel_hierarchy_parallel.py` — Distributes `pixel_hierarchy.py` across covariate × hierarchy
- `04_pixel_urban_main_parallel.py` — Distributes `pixel_urban_main.py` across threshold × hierarchy × block
- `05_pixel_urban_hierarchy_parallel.py` — Distributes `pixel_urban_hierarchy.py` across threshold × hierarchy

---

## Stage 02: `02_data_prep/` — Data Preparation

**Purpose:** Build hierarchies, process population, rake outcomes to GBD, create modeling and forecast dataframes.

### Files

#### `00_make_covariate_means.py`
- **What it does:** Aggregates climate, urbanization, income, and DAH covariates by SSP scenario; exports as xarray NetCDF.
- **Inputs:** LSAE pop CSVs; flooding, urban, income, climate, DAH, hierarchy parquets
- **Outputs:** `04-forecasting_data/covariate_means.nc`
- **Functions:** `read_parquet_with_integer_ids()`, `write_netcdf()`, `read_urban_paths()`, `read_income_paths()`, `merge_dataframes()`
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`, `helper_functions`

#### `01_make_full_hierarchy.py`
- **What it does:** Merges GBD, FHS, LSAE hierarchies into one comprehensive hierarchy with cross-walk tables.
- **Inputs:** LSAE, GBD, FHS hierarchy parquets
- **Outputs:** `02-processed_data/full_hierarchy_2023_lsae_1209.parquet/.nc`; `lsae_to_fhs_table.parquet`; `lsae_to_gbd_table.parquet`
- **Functions:** `read_parquet_with_integer_ids()`, `write_parquet()`, `write_netcdf()`, `convert_to_xarray()`
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### `02_as_fhs_and_full_population.py`
- **What it does:** Processes all-age and age-specific population data from LSAE and FHS; rakes to align estimates; produces harmonized 2000–2100 population.
- **Inputs:** GBD/FHS population parquets and netCDF; LSAE population CSVs; hierarchy; age metadata
- **Outputs:** `02-processed_data/aa_2023_full_population.parquet/.nc`; `as_2023_full_population.parquet/.nc`; FHS variants; `age_sex_df.parquet`; missing location parquets
- **Functions:** `read_parquet_with_integer_ids()`, `write_parquet()`, `write_netcdf()`, `convert_with_preset()`, `level_filter()`
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### `03_rake_aa_A2_to_GBD.py`
- **What it does:** Rakes all-age malaria and dengue LSAE estimates to GBD totals; computes rates; merges metrics.
- **Inputs:** Population, hierarchy, GBD reference CSVs, LSAE cause CSV files
- **Outputs:** `02-processed_data/aa_full_malaria_df.parquet/.nc`; `aa_full_dengue_df.parquet/.nc`
- **Functions:** `process_lsae_df()`, `format_aa_gbd_df()`, `rake_aa_count_lsae_to_gbd()`, `make_aa_full_rate_df_from_aa_count_df()`, `aggregate_aa_rate_lsae_to_gbd()`, `check_concordance()`
- **Project imports:** `constants`, `cause_processing_functions`, `rake_and_aggregate_functions`, `parquet_functions`, `xarray_functions`, `helper_functions`

#### `04_rake_as_A2_to_GBD.py`
- **What it does:** Creates age-specific disease datasets by disaggregating all-age LSAE counts using GBD age-specific rate ratios.
- **Inputs:** Hierarchy, age-sex combinations, age-specific population, GBD age-specific data, all-age disease data
- **Outputs:** `02-processed_data/as_full_{cause}_ds.nc/.parquet` for each cause
- **Functions:** `read_parquet_with_integer_ids()`, `write_parquet()`, `level_filter()`, `convert_to_xarray()`, `write_netcdf()`
- **Project imports:** `constants`, `parquet_functions`, `helper_functions`, `xarray_functions`

#### `05_malaria_modeling_dataframe.py`
- **What it does:** Prepares all-age malaria modeling dataset; filters to most-detailed locations; merges covariates; applies log/logit transforms; creates base/age-specific modeling datasets.
- **Inputs:** Hierarchy, age-sex metadata, malaria all-age and age-specific parquets, covariate parquets (income, urban, climate, DAH)
- **Outputs:** `03-modeling_data/aa_ge3_malaria_stage_1_modeling_df.parquet`, `aa_md_malaria_pfpr_modeling_df.parquet`, `as_md_malaria_modeling_df.parquet`, `base_md_malaria_modeling_df.parquet`, `rest_md_malaria_modeling_df.parquet`
- **Functions:** `read_parquet_with_integer_ids()`, `write_parquet()`, `level_filter()`, `read_urban_paths()`, `read_income_paths()`, `merge_dataframes()`
- **Project imports:** `constants`, `parquet_functions`, `helper_functions`

#### `06_dengue_modeling_dataframe.py`
- **What it does:** Parallel to `05_malaria_modeling_dataframe.py` for dengue; also computes case fatality ratios.
- **Inputs:** Hierarchy, dengue all-age and age-specific parquets, covariate parquets
- **Outputs:** `03-modeling_data/aa_ge3_dengue_stage_1_modeling_df.parquet`, `as_md_dengue_modeling_df.parquet`, base/rest datasets
- **Functions:** `read_parquet_with_integer_ids()`, `write_parquet()`, `read_urban_paths()`, `read_income_paths()`, `merge_dataframes()`
- **Project imports:** `constants`, `parquet_functions`, `helper_functions`

#### `07_forecasted_dataframes_non_draw_part.py`
- **What it does:** Creates fixed-effects (non-draw-varying) forecast dataset components by merging population, hierarchy, flooding, urban, income, DAH data by SSP scenario.
- **Inputs:** Flooding parquets, urban/income parquets, DAH parquet, population, hierarchy
- **Outputs:** `04-forecasting_data/malaria_forecast_scenario_{ssp}_non_draw_part.parquet`; dengue equivalent
- **Functions:** `read_parquet_with_integer_ids()`, `write_parquet()`, `read_urban_paths()`, `read_income_paths()`, `merge_dataframes()`, `ensure_id_columns_are_integers()`, `sort_id_columns()`
- **Project imports:** `constants`, `parquet_functions`, `helper_functions`

#### `forecasted_draw_specific_malaria_dataframes.py`
- **What it does:** Creates draw-specific malaria forecast files; generates DAH scenarios (Baseline, Constant, Increasing, Decreasing) with multiplicative factors; merges draw-specific climate data.
- **Inputs:** CLI: `--ssp_scenario`, `--draw`; non-draw baseline parquet; hierarchy; climate draw parquets
- **Outputs:** Four DAH-scenario parquets per (ssp, draw): Baseline, Constant, Increasing, Decreasing
- **Functions:** `generate_dah_scenarios()` — creates DAH variants with factors [1.2, 1.4, 1.6, 1.8, 2.0] (increasing) and [0.8, 0.6, 0.4, 0.2, 0.0] (decreasing)
- **Project imports:** `constants`, `parquet_functions`, `helper_functions`

#### `forecasted_draw_specific_dengue_dataframes.py`
- **What it does:** Creates draw-specific dengue forecast files; filters to high-incidence locations; merges draw-specific climate data; computes CFR and log transforms.
- **Inputs:** CLI: `--ssp_scenario`, `--draw`; non-draw baseline parquet; hierarchy; age-specific dengue data; climate draw parquets
- **Outputs:** `04-forecasting_data/dengue_forecast_ssp_scenario_{ssp}_draw_{draw}.parquet`
- **Functions:** `read_parquet_with_integer_ids()`, `write_parquet()`, `level_filter()`
- **Project imports:** `constants`, `parquet_functions`, `helper_functions`

#### `BG_forecasted_malaria_dataframes.py` / `alt_forecasted_malaria_dataframes.py`
- **What they do:** Create alternative malaria forecasting datasets with Goalkeepers DAH scenarios (GK_reference, GK_cut20); read baseline and merge new DAH by country.
- **Inputs:** Baseline forecast parquets; `GK_dah_ref_df_2025_07_08.parquet`; `GK_dah_cut20_df_2025_07_08.parquet`
- **Outputs:** GK_reference and GK_cut20 scenario parquets
- **Project imports:** `constants`, `parquet_functions`

#### `forecasted_draw_specific_malaria_dataframes_old.py`
- **What it does:** Archived prior version of malaria forecasting draw script. Retained for reference.

#### Orchestrators
- `08_forecasted_malaria_dataframes_parallel.py` — Distributes malaria draw dataframe creation across ssp × draw (10,000 tasks)
- `09_forecasted_dengue_dataframes_parallel.py` — Distributes dengue draw dataframe creation
- `88_alt_forecasted_malaria_dataframes_parallel.py` — Distributes alternative DAH scenario creation

---

## Stage 03: `03_modeling/` — Statistical Modeling

**Purpose:** Fit regression models for malaria and dengue.

**Python files: NONE.** This directory contains only R scripts (13 `.r` files) and PDF reference documents. All modeling is done in R. Do not touch without explicit permission.

---

## Stage 04: `04_forecasting/` — Apply Models and Disaggregate Age-Sex

**Purpose:** Apply fitted models to forecast data; rake dengue to observed totals; disaggregate all-age to age-sex.

### Files

#### `as_malaria_fractions.py`
- **What it does:** Allocates all-age malaria mortality and incidence to age-sex groups using relative risk-weighted fractions; excludes infants (age_group_id=2).
- **Inputs:** CLI: `--ssp_scenario`, `--dah_scenario`, `--draw`, `--hold_variable`; malaria forecast netCDF (or parquet); hierarchy, population, RR reference, cause data parquets
- **Outputs:** Two netCDF files: incidence and mortality age-sex splits
- **Functions:** `log_time_and_memory()`
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### `as_malaria_shifts.py`
- **What it does:** Disaggregates all-age malaria forecasts into age-sex-specific estimates using RR patterns; calculates predicted incidence and mortality counts.
- **Inputs:** CLI: `--ssp_scenario`, `--dah_scenario`, `--draw`; malaria predictions parquet; hierarchy, population, RR reference parquets
- **Outputs:** Two parquets: incidence measures and mortality measures with age-sex predictions
- **Functions:** `log_time_and_memory()`
- **Project imports:** `constants`, `parquet_functions`

#### `rake_dengue.py`
- **What it does:** Rakes all-age dengue incidence to 2022 observed; applies age-sex CFR adjustments from regression coefficients; outputs raked predictions.
- **Inputs:** CLI: `--ssp_scenario`, `--draw`, `--hold_variable`; dengue forecast parquet or netCDF; hierarchy, population, cause data; `as_id_levels.csv`; `mod_cfr_all_coefficients.csv`
- **Outputs:** netCDF with raked dengue incidence and CFR predictions
- **Functions:** None defined
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### `as_dengue_shifts.py`
- **What it does:** Disaggregates raked dengue to age-sex groups; applies vaccination effects for selected locations; calculates mortality via CFR.
- **Inputs:** CLI: `--ssp_scenario`, `--draw`, `--hold_variable`; raked dengue netCDF; age-sex population, RR reference, vaccine data, hierarchy parquets
- **Outputs:** Two netCDF files: incidence and mortality age-sex splits (plus no-vaccine variants)
- **Functions:** None defined
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### Orchestrators
- `03_as_malaria_fractions_parallel.py` — Distributes `as_malaria_fractions.py` across ssp × dah × draw
- `04_rake_dengue_parallel.py` — Distributes `rake_dengue.py` across ssp × draw
- `05_as_dengue_shifts_parallel.py` — Distributes `as_dengue_shifts.py` (with optional hold-out variables)
- `96_as_malaria_shifts_parallel.py` — Distributes `as_malaria_shifts.py` across ssp × dah × draw

---

## Stage 05: `05_aggregation/` — Hierarchy Aggregation and Raking

**Purpose:** Aggregate admin-2 predictions up through geographic hierarchy; rake to FHS totals; create DALYs; create hold-variable sensitivity datasets.

### Files

#### `cause_as_aggregation_by_draw.py`
- **What it does:** Aggregates forecast data from leaf (level-5) locations up through hierarchy; no raking.
- **Inputs:** CLI: `--cause`, `--measure`, `--ssp_scenario`, `--dah_scenario`, `--draw`, `--hold_variable`, `--run_date`; forecast netCDF; hierarchy parquet
- **Outputs:** netCDF → `{UPLOAD_DATA_PATH}/upload_folders/{run_date}/full_as_{cause}_measure_{measure}_...`
- **Functions:** `process_forecast_data(forecast_ds_path, measure, hierarchy_df)`
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### `cause_as_aggregation_by_draw_raked.py`
- **What it does:** Like `cause_as_aggregation_by_draw.py` but applies FHS raking correction factors at leaf locations before aggregating.
- **Inputs:** Same as above plus raked admin-2 netCDF from `create_raked_outcomes.py` outputs
- **Outputs:** netCDF with raking-corrected aggregated values
- **Functions:** `process_forecast_data(forecast_ds_path, measure, hierarchy_df, ratio_ds)` — ratio_ds carries correction factors
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### `create_raked_outcomes.py`
- **What it does:** Rakes admin-2 predictions to match FHS admin-1 totals; handles deprecated location ID remapping.
- **Inputs:** CLI: `--cause`, `--scenario`, `--measure`, `--draw`; FHS netCDF; admin-2 predictions netCDF; full hierarchy netCDF
- **Outputs:** Raked netCDF → `…/output/2025_09_08/as_cause_{cause}_measure_{measure}_…_raked/draw_{draw}.nc`
- **Functions:** `impute_location_ids()`, `load_draws()`, `get_forcasted_ds()`, `get_predicted_ds()`, `load_in_full_hierarchy_dataset()`, `sum_and_align_admin2_totals()`, `broadcast_factor_to_admin2()`, `build_raked_dataset()`, `save_raked_dataset_optimized()`, `main_raking_function()`
- **Project imports:** `constants`, `xarray_functions`

#### `create_as_dalys_by_draw_raked_parallel.py` (core script)
- **What it does:** Combines raked YLD and YLL datasets to produce DALYs per draw.
- **Inputs:** CLI: `--cause`, `--ssp_scenario`, `--dah_scenario`, `--draw`, `--hold_variable`, `--run_date`; YLD and YLL netCDF files
- **Outputs:** DALY netCDF file
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### `make_population_hold_variables_by_draw.py`
- **What it does:** Creates hold-variable sensitivity scenarios by scaling base forecasts to reference-year population distributions.
- **Inputs:** CLI: `--cause`, `--measure`, `--ssp_scenario`, `--draw`, `--run_date`; base forecast netCDF; age-specific and all-age population netCDF; hierarchy netCDF
- **Outputs:** Two netCDF files per draw: population-hold and as_structure-hold variants
- **Functions:** `get_draw_path()`, `process_forecast_data(ds, hierarchy_df)`
- **Project imports:** `constants`, `parquet_functions`, `xarray_functions`

#### Orchestrators
- `01_cause_as_aggregation_by_draw_parallel.py` — Distributes non-raked aggregation
- `02_create_raked_outcomes_parallel.py` — Distributes raking (with `check_if_path_draw_exists()` skip logic)
- `03_cause_as_aggregation_by_draw_raked_parallel.py` — Distributes raked aggregation
- `04_create_as_dalys_by_draw_raked_parallel.py` — Distributes DALY creation
- `05_make_population_hold_variables_by_draw_parallel.py` — Distributes hold-variable creation

---

## Stage 06: `06_upload/` — Draw Combination and Upload Preparation

**Purpose:** Combine 100 per-draw files into draw-stacked datasets; compute means; produce AA summary; format for FHS/Goalkeepers upload.

### Files

#### `create_and_combine_as_and_aa_draws.py`
- **What it does:** Combines all per-draw forecast files into one dataset; adds sex=3 aggregate; reindexes to full coordinate space; aggregates to AA; computes rates; writes draws and mean files for both AS and AA.
- **Inputs:** 100 per-draw netCDF files; age-specific population netCDF; age metadata parquet; hierarchy parquet
- **Outputs:** `draws.nc` and `mean.nc` for AS and AA under `{UPLOAD_DATA_PATH}/upload_folders/{run_date}/`
- **Functions:** `get_as_path()`, `get_draw_file_path()`, `calculate_rate_and_mean(ds_raw, metric_type, ds_type)`
- **Project imports:** `constants`, `helper_functions`, `hd5_functions`, `parquet_functions`, `xarray_functions`

#### `create_and_combine_as_draws.py`
- **What it does:** Like above but AS-only output (AA created but not written).
- **Inputs/Outputs:** Same structure as above
- **Functions:** `get_as_path()`, `get_draw_file_path()`, `calculate_rate_and_mean()`
- **Project imports:** same as above

#### `create_and_combine_aa_draws.py`
- **What it does:** Like above but AA-only output.
- **Inputs/Outputs:** Same structure
- **Functions:** `get_aa_path()`, `get_draw_file_path()`
- **Project imports:** same as above

#### `combine_as_draws.py`
- **What it does:** Combines per-draw files with three output modes: standard netCDF, FHS HDF5 format, Goalkeepers HDF5 format. Handles Ethiopian location swaps.
- **Inputs:** CLI with `--fhs_flag`; per-draw netCDF; population parquet or netCDF; age metadata; hierarchy parquet
- **Outputs:** Standard: `draws.nc`; FHS: `draws.h5` with FHS metadata; GK: `draws.h5` with Goalkeepers metadata
- **Functions:** `get_draw_file_path()`
- **Project imports:** `constants`, `hd5_functions`, `parquet_functions`, `xarray_functions`

#### `create_summary_files.py`
- **What it does:** Aggregates AS count draws across age/sex to AA; computes mean counts and rates; outputs all-age summary files.
- **Inputs:** Count draw netCDF files from upload folders
- **Outputs:** `all_age_draws.nc` and `all_age_mean.nc` for AS count, AS rate, AA count, AA rate
- **Project imports:** `constants`, `xarray_functions`

#### `04_make_full_aa_ds.py`
- **What it does:** Combines historical disease data with future projections for AA datasets; merges population and hierarchy; calculates rates; writes consolidated netCDF by DAH scenario.
- **Inputs:** Historical AA netCDF; population and hierarchy netCDF; future forecast means from upload folders
- **Outputs:** `05-upload_data/upload_folders/{run_date}/full_aa_ds_{dah_scenario}.nc`
- **Project imports:** `constants`, `xarray_functions`, `hd5_functions`, `parquet_functions`

#### `05_make_full_as_ds.py`
- **What it does:** Parallel to `04_make_full_aa_ds.py` for age-sex-specific datasets.
- **Inputs/Outputs:** AS equivalents of above
- **Project imports:** `constants`, `xarray_functions`, `hd5_functions`, `parquet_functions`

#### Orchestrators
- `01_combine_as_draws_parallel.py` — Distributes `combine_as_draws.py` (supports fhs_flag for multi-format output)
- `03_create_and_combine_aa_draws_parallel.py` — Distributes `create_and_combine_aa_draws.py`
- `06_create_and_combine_as_and_aa_draws_parallel.py` — Distributes `create_and_combine_as_and_aa_draws.py` (metrics × measures)
- `99_create_summary_files_parallel.py` — Distributes `create_summary_files.py`

---

## Utility Modules: `src/idd_forecast_mbp/*.py`

### `constants.py`
- **What it does:** Single source of truth for all paths, lookup maps, scenario definitions, and shared configuration. No functions — just constants.
- **Key contents:** `MODEL_ROOT`, `REPO_ROOT`, all stage data paths; `draws` (list of 100 draw strings `"000"`–`"099"`); `cause_map`, `measure_map`, `metric_map`, `full_measure_map`, `ssp_scenarios`, `ssp_scenario_map`; `aa_merge_variables`, `as_merge_variables`; `malaria_variables`, `dengue_variables` (raw data paths); `modeling_measure_map`; `FHS_RESULTS_PATH`, `fhs_population_paths`, `fhs_draws`; `covariate_map`
- **Imported by:** Every single Python file in the project, always as `import constants as rfc`

---

### `parquet_functions.py`
- **What it does:** Read and write parquet files with retry logic, validation, integer ID coercion, and consistent sort order.
- **Functions:**
  - `read_parquet_with_integer_ids(path, **kwargs)` — Read parquet, sort by ID columns, cast `*_id` columns to integer
  - `write_parquet(df, filepath, ...)` — Write parquet with configurable validation (none/metadata/sample/full), optional atomic write, retry, 775 permissions; defaults to `lz4` compression
  - `ensure_id_columns_are_integers(df)` — Cast all `*_id` float columns to `Int64`
  - `sort_id_columns(df)` — Sort by `location_id`, `year_id`, then other `*_id` columns
  - `filter_df(df, **id_filters)` — Filter DataFrame by one or more ID columns (single value or list)
  - `filter_df_by_range(df, **column_ranges)` — Filter by (min, max) tuple per column
- **Imported by:** Stages 02, 04, 05, 06 (universally); also `helper_functions`, `covariate_functions`, `fhs_functions`, `counterfactual_functions`, `data_functions`, `color_functions`, `bin_functions`, `map_functions`

---

### `xarray_functions.py`
- **What it does:** Read and write netCDF/xarray datasets with retry, compression, chunking, integer coordinate coercion, and DataFrame↔xarray conversion.
- **Functions:**
  - `read_netcdf_with_integer_ids(path, **kwargs)` — Open dataset, cast coordinates, sort
  - `write_netcdf(ds, filepath, ...)` — Write netCDF with configurable compression (zlib), chunking (auto/manual/by-dim), temp-file atomic write, dimension/variable shape validation, 775 permissions
  - `convert_to_xarray(df, dimensions=None, ...)` — Convert DataFrame to xarray Dataset; auto-detects `*_id` dimensions, auto-optimizes dtypes, validates rectangular grid
  - `convert_with_preset(df, preset='as_variables', ...)` — Shortcut using `COMMON_DIMENSION_CONFIGS` presets: `'as_variables'` (location/year/age/sex) or `'aa_variables'` (location/year)
  - `sort_coordinates(ds, ...)` — Sort dataset by 1D coordinates, prioritizing `location_id`, `year_id`
  - `ensure_id_coordinates_are_integers(ds)` — Cast all `*_id` coords to optimal integer dtype
  - `cast_coordinate_types(ds)` — Cast standard coords to fixed dtypes (int32/int16/int8)
  - `filter_ds_by_multiple_coords(ds, **coord_filters)` — Filter dataset by multiple coordinates
  - `filter_ds_by_single_coord(ds, coord_name, coord_values)` — Filter by one coordinate
  - `filter_ds_with_sel(ds, **coord_filters)` — Filter via `.sel()`
  - `filter_ds_by_range(ds, **coord_ranges)` — Filter by (min, max) per coordinate
  - `get_unique_coords(ds, coord_names=None)` — Return dict of unique values per coordinate
  - `optimize_netcdf_encoding(ds, compression_level=4)` — Build encoding dict for manual use
  - `print_netcdf_dimensions(filepaths)` — Diagnostic: print dim sizes for a list of files
  - Various private helpers: `_fix_nullable_dtypes`, `_ensure_xarray_compatible_dtypes`, `_auto_optimize_dimension_dtype`, `_auto_optimize_variable_dtype`, `_validate_rectangular_grid`
- **Imported by:** Stages 02, 04, 05, 06; also `covariate_functions`, `fhs_functions`, `counterfactual_functions`, `data_functions`, `color_functions`, `bin_functions`, `map_functions`

---

### `helper_functions.py`
- **What it does:** Mid-level data loading and manipulation helpers used across pipeline stages.
- **Functions:**
  - `merge_dataframes(model_df, dfs)` — Left-merge a dict of DataFrames onto a base frame on `['location_id', 'year_id']`
  - `read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH)` — Load income parquets filtered to one RCP scenario
  - `read_urban_paths(urban_paths, VARIABLE_DATA_PATH)` — Load urban parquets with column renaming normalization
  - `level_filter(hierarchy_df, start_level, end_level=None, return_ids=False)` — Build a pyarrow filter tuple for hierarchy levels; optionally return location IDs
  - `check_folders_for_files(folders_and_files, delete_existing=True)` — Check and optionally delete files from multiple folders; creates folders if missing
  - `check_column_for_problematic_values(column_name, df, ...)` — Check for NaN, inf, negative, non-numeric values with optional verbose report
  - `verify_hdf_checksum(filepath, key='df')` — Verify HDF5 file integrity via stored SHA-256 checksum
- **Imported by:** Stages 01 (`pixel_hierarchy.py`, `pixel_urban_hierarchy.py`), 02 (multiple), 04, 06; also `cause_processing_functions`, `color_functions`, `bin_functions`
- **Note:** Module-level code sets `VARIABLE_DATA_PATH = PROCESSED_DATA_PATH / 'lsae_1209'` — this hardcodes hierarchy version

---

### `hd5_functions.py`
- **What it does:** Read and write HDF5 files for final FHS/Goalkeepers upload format.
- **Functions:**
  - `write_hdf(df, filepath, key='df', ...)` — Write DataFrame to HDF5 with blosc:zstd compression, validation, retry, 775 permissions
  - `benchmark_hdf_compression(df, filepath_base, ...)` — Benchmark multiple compression options and return summary
  - `create_hdf_structure(file_path, metadata_df, draw_columns, metadata_columns)` — Pre-allocate HDF5 structure with metadata and draw columns (uses h5py directly)
  - `write_draw_column(file_path, draw_column, values)` — Write values to a specific draw column in existing HDF5
  - `read_hdf_metadata(file_path, metadata_columns)` — Read only metadata columns from HDF5
  - `append_to_hdf_table(file_path, new_data, table_name)` — Append rows to HDF5 table
  - `check_hdf_structure(file_path)` — Return dict of dataset names, shapes, dtypes
- **Imported by:** Stage 06 (`combine_as_draws.py`, `create_and_combine_*.py`, `04_make_full_aa_ds.py`, `05_make_full_as_ds.py`); also `counterfactual_functions.py`, `save_functions.py`

---

### `yaml_functions.py`
- **What it does:** Load and parse the `COVARIATE_DICT.yaml` configuration file.
- **Functions:**
  - `load_yaml_dictionary(yaml_path)` — Load YAML and return the `COVARIATE_DICT` key
  - `parse_yaml_dictionary(covariate)` — Load YAML, extract config for a specific covariate; returns dict with `covariate_name`, `covariate_resolution`, `years`, `synoptic`, `cc_sensitive`, `summary_statistic`, `path`
- **Imported by:** Stage 01 (`01_prep_maps.py`, `pixel_main.py`, `pixel_urban_main.py`, orchestrators); also `pixel_hierarchy.py`, `pixel_urban_hierarchy.py` (import as `helper_functions.parse_yaml_dictionary` — **likely a bug**)

---

### `rake_and_aggregate_functions.py`
- **What it does:** Core raking and aggregation logic to reconcile LSAE admin-2 estimates with GBD national totals.
- **Functions:**
  - `rake_aa_count_lsae_to_gbd(...)` — Main entry point: rakes level-4 then level-5 LSAE counts to match GBD level-3 targets; handles problematic raking factors by switching to population-based raking
  - `rake_level(count_variable, level_df, level_m1_df, problematic_rules, hierarchy_df, level)` — Single-level raking step; computes count-based and population-based factors; tracks and flags problematic rows
  - `aggregate_aa_count_lsae_to_gbd(...)` — Pure aggregation (no raking): sums level-5 up to all higher levels
  - `aggregate_level(count_variable, level_df, hierarchy_df)` — Single-level sum aggregation via parent_id
  - `aggregate_aa_rate_lsae_to_gbd(...)` — Convert rates to counts, aggregate, convert back to rates
  - `make_aa_full_rate_df_from_aa_count_df(...)` — Compute rates from counts and population
  - `make_aa_rate_variable(...)` — Legacy rate variable computation
  - `make_aa_df_square(variable, df, hierarchy_df, ...)` — Fill missing location×year combinations with zeros to ensure complete rectangular grid
  - `prep_df(df, hierarchy_df)` — Add `level` column from hierarchy, remove `parent_id`
- **Imported by:** Stage 02 (`03_rake_aa_A2_to_GBD.py`); also `cause_processing_functions.py`

---

### `cause_processing_functions.py`
- **What it does:** Format and process raw GBD and LSAE disease data for malaria and dengue.
- **Functions:**
  - `format_aa_gbd_df(cause, measure, metric, df, ...)` — Filter and rename GBD DataFrame to standardized variable names
  - `process_lsae_df(cause, measure, aa_full_population_df, hierarchy_df)` — Load LSAE per-capita data, compute counts, fill missing locations, merge population; handles malaria-specific `pfpr` vs count logic
  - `check_concordance(variable, aa_full_df, aa_gbd_df, tolerance=0.01)` — Compare two DataFrames and report concordance statistics
  - `drop_scenario_population_mean(df)` — Drop scenario, population, and mean columns; convenience wrapper
  - `rename_columns(df)` — Normalize column names (strip `_mean_per_capita`, fix `rate_mean` → `count`)
- **Imported by:** Stage 02 (`03_rake_aa_A2_to_GBD.py`) only

---

### `covariate_functions.py`
- **What it does:** Clean xarray-based loaders for all covariates (suitability, income, flooding, urban, DAH). Newer interface that returns xarray Datasets with scenario dimensions.
- **Functions:**
  - `get_cause_suitability_df(cause, year_ids, location_ids, ssp_scenario)` — Load draw-level suitability parquet for one scenario
  - `get_cause_suitability_ds(cause, year_ids, location_ids, ssp_scenarios)` — Load suitability across scenarios, return xarray Dataset with `ssp_scenario` and `draw` dimensions
  - `get_income_ds(year_ids, location_ids)` — Load GDP per capita xarray with `ssp_scenario` dimension
  - `get_flooding_ds(year_ids, location_ids, ssp_scenarios)` — Load flooding data xarray with `ssp_scenario` dimension
  - `get_urban_ds(year_ids, location_ids)` — Load 1km urban threshold-300 data as xarray
  - `get_dah_ds(year_ids, location_ids, dah_scenarios)` — Load DAH data xarray with `dah_scenario` dimension
  - `convert_draws_to_xarray(df)` — Melt wide-format draw columns to long format and convert to xarray
- **Imported by:** `data_functions.py` (star import); stage 02 appears to use the older `read_*_paths` approach from `helper_functions` instead

---

### `fhs_functions.py`
- **What it does:** Load FHS (Forecasting Health Scenarios) external data for comparison.
- **Functions:**
  - `get_fhs_ds_multi(data_dict)` — Load FHS forecast datasets for multiple SSP scenarios; handles both draw-level and summary files; returns combined xarray Dataset
  - `get_data_dict(draws, cause, measure, metric, ...)` — Build a standardized data request dictionary
  - `build_selection_dict(data_dict)` — Convert data_dict to xarray `.sel()` dict (excluding 'all' values)
  - `get_location_id(location_name)` — Look up location_id from name using hierarchy DataFrame
- **Note:** Module-level code reads hierarchy parquet at import time
- **Imported by:** `counterfactual_functions.py`, `save_functions.py` (star imports)

---

### `number_functions.py`
- **What it does:** Number formatting for uncertainty intervals in manuscript text.
- **Functions:**
  - `get_value_w_UI(vec, ...)` — Format a vector of draws as "mean (lower–upper)" with 3 sig figs, optional millions/billions scaling, percentage or rate mode
  - `smart_UI_format(val, ...)` — Format a single number at 3 sig figs with optional scaling
  - `get_multiplier(number, ...)` — Return scale factor and label (e.g., `0.000001`, `" (in Millions)"`) for a given magnitude
  - `get_summary_ds_from_ds(ds, ...)` — Compute mean/lower/upper (2.5%/97.5%) xarray Dataset from draws
  - `get_mean_UI_matrix(df)` — Row-wise mean, 2.5th, 97.5th percentile summary DataFrame
  - `get_value_w_UI` aliases: `get_UI_text()`, `format_mean_lower_upper()`
- **Imported by:** `counterfactual_functions.py`, `save_functions.py`, `plot_functions.py` (star imports or specific)

---

### `counterfactual_functions.py`
- **What it does:** Load and compare reference vs. alternative scenario forecast datasets for analysis/visualization. **⚠️ Nearly identical to `save_functions.py`** — both define `get_aa_path`, `load_ref_and_alt_ds`, `calculate_dss_and_statistics`, `print_short_dict`.
- **Functions:**
  - `get_aa_path(cause, measure, metric, ssp_scenario, ...)` — Build upload folder path for AS and AA datasets
  - `load_ref_and_alt_ds(...)` — Load reference and alternative scenario netCDF files; return combined data dict
  - `calculate_dss_and_statistics(data_dict, location_id, ...)` — Compute difference, relative difference, cumulative versions; return plot-ready dict with summary statistics
  - `print_short_dict(dict)` — Pretty-print a subset of keys from a data dict
- **Imported by:** `data_functions.py` (star import)
- **⚠️ DUPLICATION:** Functions are copy-pasted from or into `save_functions.py`; `get_aa_path` also exists identically in `06_upload/create_and_combine_as_and_aa_draws.py` variants

---

### `save_functions.py` (misnamed)
- **What it does:** Despite its name, this module has two distinct halves: (1) matplotlib figure-saving utilities, and (2) scenario comparison and analysis functions that duplicate `counterfactual_functions.py`.
- **Functions:**
  - `save_figure_as_pdf(fig, filename_base, ...)` — Save figure as PDF; optional thumbnail
  - `save_figure_as_png(fig, filename_base, ...)` — Save figure as PNG
  - `save_figure_as_pdf_and_png(fig, filename_base, ...)` — Save both formats
  - `save_thumbnail_figure_as_png_1/2(fig, thumbnail, ...)` — Create thumbnail by pixel-shrinking or by scaling figure size
  - `get_aa_path(...)` — **Duplicate** of `counterfactual_functions.get_aa_path`
  - `load_ref_and_alt_ds(...)` — **Duplicate** of `counterfactual_functions.load_ref_and_alt_ds`
  - `calculate_dss_and_statistics(...)` — **Duplicate** of `counterfactual_functions.calculate_dss_and_statistics`
  - `print_short_dict(dict)` — **Duplicate**
- **Imported by:** `map_functions.py` (star import for figure-saving functions)

---

### `data_functions.py`
- **What it does:** Load data for map plotting — polygons, raster data, admin-2 outcome data.
- **Functions:**
  - `read_polygons()` — Load admin-0/1/2 and disputed area shapefiles from standard paths
  - `load_population_data(year, resolution)` — Load population GeoTIFF for a given year/resolution
  - `load_suitability_raster_data(map_plot_dict)` — Load suitability netCDF raster data for map periods
  - `load_cov_data(cov_dict, ...)` — Load flood or storm raster data for a year/scenario
  - `get_raster_data(map_plot_dict)` — Orchestrate raster data loading for all map periods
  - `get_admin2_data(map_plot_dict)` — Load admin-2 forecast outcomes and merge with location info
  - `get_outcome_df(map_plot_dict)` — Load and compute outcome DataFrame for mapping (handles all measure types)
  - `update_loc_ids(map_plot_dict)` — Populate location ID lists for endemic vs. all locations
  - `make_cov_count(outcome_ds)` — Multiply covariate rates by population for count totals
  - `reproject_cov_slice(cov_layer, population_data, transform)` — Reproject covariate raster to population grid via nearest-neighbor interpolation
- **Note:** Module-level code reads hierarchy, population, and covariate datasets at import time; hardcodes `best_run_date = "2025_08_11"`
- **Imported by:** `map_functions.py`

---

### `color_functions.py`
- **What it does:** Create colormaps and bin-color assignments for map plots.
- **Functions:**
  - `get_colors(n_bins, cmap_name='Reds')` — Sample n_bins colors from a matplotlib colormap
  - `create_outcome_colormap(plot_dict)` — Build listed colormap for absolute outcome maps (single-direction)
  - `create_change_colormap(plot_dict, clip_neg, clip_pos)` — Build diverging colormap for change maps; supports removing middle color, white center
  - `create_diverging_colors(n_bins, cmap_name='RdBu_r')` — Generate n_bins diverging colors
- **Imported by:** `bin_functions.py`, `plot_functions.py`, `data_functions.py`, `map_functions.py`

---

### `bin_functions.py`
- **What it does:** Legend bin formatting and rendering for map figures.
- **Functions:**
  - `get_bin_info(map_plot_dict, plot_data)` — Assign colormap and compute categorical bin assignments for plot data
  - `draw_legend_bins(ax, map_plot_dict)` — Draw legend rectangles and labels on a matplotlib axis
  - `add_legend(fig, ax, map_plot_dict)` — Dispatch to panel or colorbar legend
  - `add_colorbar(fig, ax, map_plot_dict)` — Add horizontal colorbar to figure
  - `add_legend_panel_colorbar(fig, ax_legend, map_plot_dict)` — Add colorbar to a dedicated legend panel
  - `pretty_bin_labels(map_plot_dict)` — Generate formatted bin label strings (with ≤/≥, units, abbreviation)
  - `smart_format(val)` — Format a number as integer with commas or 2-decimal float
  - `clip_data_to_bins(df, data_column, bins)` — Clip data values to bin boundaries
- **Imported by:** `plot_functions.py`, `data_functions.py`, `map_functions.py`

---

### `plot_functions.py`
- **What it does:** Create and populate matplotlib figures for time series and map plots.
- **Key functions (first 80 lines captured):**
  - `create_figure(plot_dict)` — Create figure with layout from `layout_dict`; handles map, gridplot, or single-axis modes
  - `plot_custom_legend(...)` — Draw a custom legend panel
  - Additional map/time series plot functions (full file too large to read completely)
- **Imported by:** `map_functions.py`

---

### `map_functions.py`
- **What it does:** High-level map plot orchestrator. Takes a `plot_dict` and drives the full map creation pipeline.
- **Key functions:**
  - `create_map_plot_dict(...)` — Build a comprehensive plot configuration dictionary from ~50 parameters; sets defaults for layout, bins, colors, extents, titles
  - `plot_map(plot_dict)` — Full map rendering pipeline: load data, create figure, draw map, add legend, save
  - `get_layout_dict(map_plot_dict)` — Compute pixel positions for map and legend panels
  - `get_period_info(map_plot_dict)` — Validate and parse period/scenario combinations by map type (change, scenario_comparison, outcome, arbitrary_comparison)
  - `get_labels(map_plot_dict)` — Assign outcome labels based on measure type
  - `get_save_path(map_plot_dict)` — Generate save path; sets `make_figure=False` if file exists and `remake_figure=False`
- **Note:** Module-level code loads `bins_dictionary`, shapefiles, and covariate dataset at import time; hardcodes `best_run_date = "2025_08_11"`
- **Imported by:** Notebooks (not pipeline stages)

---

### `loading_functions.py`
- **What it does:** ⚠️ **Duplicate of `parquet_functions.py` + parts of `xarray_functions.py`**. Contains simpler/older implementations of the same IO functions (snappy instead of lz4, full read-back validation only, no atomic write option, simpler xarray write without chunking/compression). Also includes `write_netcdf_optimized` which is similar to `write_netcdf` with explicit encoding.
- **Functions:** `read_parquet_with_integer_ids`, `write_parquet`, `ensure_id_columns_are_integers`, `sort_id_columns`, `read_netcdf_with_integer_ids`, `write_netcdf`, `sort_id_coordinates`, `ensure_id_coordinates_are_integers`, `optimize_netcdf_encoding`, `write_netcdf_optimized`
- **Imported by:** Unknown — needs grep to confirm; likely notebook-only or an abandoned refactor target
- **⚠️ DUPLICATION:** All functions exist in `parquet_functions.py` and `xarray_functions.py` with richer implementations. This file appears to be a historical artifact.

---

### `data.py`
- **What it does:** `FloodingData` class with properties for standard data paths (logs, raw, processed, modeling, forecasting roots). Minimal live code — most methods are commented out.
- **Functions/Classes:** `FloodingData` class with `root`, `logs`, `log_dir()`, `raw_root`, `processed_root`, `modeling_root`, `forecasting_root`
- **Imported by:** Unknown; appears to be a scaffold for future use, not used by pipeline

---

### `cli.py`
- **What it does:** Empty Click CLI entry point. `cli()` command does nothing.
- **Imported by:** `pyproject.toml` entry point; not used in pipeline

---

### `__init__.py`
- **What it does:** Empty.

---

## Utility Module Summary

### Pipeline-critical modules (used in stages 01–06)
| Module | Role |
|--------|------|
| `constants.py` | All paths and configuration |
| `parquet_functions.py` | Parquet I/O |
| `xarray_functions.py` | NetCDF/xarray I/O + DataFrame conversion |
| `helper_functions.py` | Merge, load covariates, level filter |
| `hd5_functions.py` | HDF5 I/O (upload only) |
| `yaml_functions.py` | COVARIATE_DICT config (stage 01 only) |
| `rake_and_aggregate_functions.py` | Raking to GBD (stage 02 only) |
| `cause_processing_functions.py` | Raw disease data formatting (stage 02 only) |

### Visualization/analysis modules (notebooks/scripts, not pipeline stages)
`covariate_functions.py`, `fhs_functions.py`, `number_functions.py`, `counterfactual_functions.py`, `save_functions.py`, `data_functions.py`, `color_functions.py`, `bin_functions.py`, `plot_functions.py`, `map_functions.py`

### Problem modules
| Module | Issue |
|--------|-------|
| `loading_functions.py` | Full duplicate of `parquet_functions` + `xarray_functions`; appears unused by pipeline |
| `counterfactual_functions.py` | Duplicates `save_functions.py` (get_aa_path, load_ref_and_alt_ds, calculate_dss_and_statistics) |
| `save_functions.py` | Mixes figure-saving code with analysis code that belongs in `counterfactual_functions.py` |

---

## Cross-Cutting Observations

### Duplicated function implementations (candidate for lib/)

| Pattern | Files with separate implementations |
|---------|--------------------------------------|
| `get_bbox()` | `pixel_main.py`, `pixel_urban_main.py` |
| `load_raking_shapes()` | `pixel_main.py`, `pixel_urban_main.py` |
| `build_bounds_map()` | `pixel_main.py`, `pixel_urban_main.py` |
| `build_location_masks()` | `pixel_main.py`, `pixel_urban_main.py` |
| `aggregate_climate_to_hierarchy()` | `pixel_hierarchy.py`, `pixel_urban_hierarchy.py` |
| `load_subset_hierarchy()` | `pixel_hierarchy.py`, `pixel_urban_hierarchy.py` |
| `post_process()` | `pixel_hierarchy.py`, `pixel_urban_hierarchy.py` |
| `hierarchy_main()` | `pixel_hierarchy.py`, `pixel_urban_hierarchy.py` |
| `process_forecast_data()` | `cause_as_aggregation_by_draw.py`, `cause_as_aggregation_by_draw_raked.py`, `make_population_hold_variables_by_draw.py` |
| `get_draw_file_path()` | `create_and_combine_as_and_aa_draws.py`, `create_and_combine_as_draws.py`, `create_and_combine_aa_draws.py`, `combine_as_draws.py` |
| `calculate_rate_and_mean()` | `create_and_combine_as_and_aa_draws.py`, `create_and_combine_as_draws.py`, `create_and_combine_aa_draws.py` |
| `generate_dah_scenarios()` | `forecasted_draw_specific_malaria_dataframes.py`, `forecasted_draw_specific_malaria_dataframes_old.py` |
| `read_urban_paths()` | `00_make_covariate_means.py`, `05_malaria_modeling_dataframe.py`, `06_dengue_modeling_dataframe.py`, `07_forecasted_dataframes_non_draw_part.py` |
| `read_income_paths()` | same 4 files |
| `merge_dataframes()` | same 4 files |
| `level_filter()` | `02_as_fhs_and_full_population.py`, `04_rake_as_A2_to_GBD.py`, `05_malaria_modeling_dataframe.py`, `forecasted_draw_specific_dengue_dataframes.py` |
| `log_time_and_memory()` | `as_malaria_fractions.py`, `as_malaria_shifts.py` |

### Shared project module imports (all stages)
- `constants` (as `rfc`) — used in every file in every stage
- `parquet_functions` — used in stages 02, 04, 05, 06
- `xarray_functions` — used in stages 02, 04, 05, 06
- `helper_functions` — used in stages 01, 02, 04, 06
- `rake_and_aggregate_functions` — used in stage 02 only (so far)
- `cause_processing_functions` — used in stage 02 only (so far)
- `yaml_functions` — used in stage 01 only
- `hd5_functions` — used in stage 06 only

### Notes
- `constants` is always imported as `rfc` — this is the alias the refactor will change to `mbpc`
- Stage 03 is R-only — do not modify
- Many orchestrators hardcode `run_date` strings (e.g. `'GK_2025_11_02'`, `'2025_08_28'`) — these are not shared constants
- `pixel_main.py` and `pixel_urban_main.py` have a typo in the CLI arg: `--hiearchy` (not `--hierarchy`) — do not fix without explicit permission
- `create_raked_outcomes.py` uses hardcoded FHS data paths outside the `constants` module
