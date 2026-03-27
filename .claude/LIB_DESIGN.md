# Lib Design
Generated: 2026-03-27
Phase 2, Task 2.2

Function signatures for `src/idd_forecast_mbp/lib/`. No implementation yet — signatures only.

Each entry lists: signature, one-line description, source file(s), and any behavior changes from the original.

---

## lib/io/parquet.py

Consolidates `parquet_functions.py` (canonical) and supersedes `loading_functions.write_parquet`.

```python
def read_parquet_with_integer_ids(
    path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """Read parquet file, sort by ID columns, and cast *_id columns to integer.

    Extracted from: parquet_functions.py:45
    """

def write_parquet(
    df: pd.DataFrame,
    filepath: str | Path,
    max_retries: int = 3,
    validate: bool = True,
    validation_method: str = 'metadata',  # 'none' | 'metadata' | 'sample' | 'full'
    overwrite: bool = True,
    compression: str = 'lz4',
    index: bool = False,
    use_atomic: bool = True,              # CHANGED: was False; now matches write_netcdf behavior
    row_group_size: int = 100000,
    **kwargs,
) -> bool:
    """Write parquet with retry, atomic rename, chmod 0o775, and optional validation.

    Always creates parent directories (os.makedirs). Supersedes loading_functions.write_parquet.
    Extracted from: parquet_functions.py:52
    Behavior change: use_atomic now defaults to True.
    """

def filter_df(
    df: pd.DataFrame,
    **id_filters,   # column_name=value_or_list
) -> pd.DataFrame:
    """Filter DataFrame by one or more ID columns. Accepts scalar or list values.

    Extracted from: parquet_functions.py:177
    """

def filter_df_by_range(
    df: pd.DataFrame,
    **column_ranges,   # column_name=(min_val, max_val)
) -> pd.DataFrame:
    """Filter DataFrame by inclusive value ranges on one or more columns.

    Extracted from: parquet_functions.py:198
    """

def ensure_id_columns_are_integers(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all *_id columns from float to Int64 (nullable integer).

    Extracted from: parquet_functions.py:14
    """

def sort_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by *_id columns; location_id and year_id sort first.

    Extracted from: parquet_functions.py:23
    """
```

---

## lib/io/netcdf.py

Consolidates `xarray_functions.py` (canonical) and supersedes `loading_functions.write_netcdf`.

```python
def read_netcdf_with_integer_ids(
    path: str | Path,
    **kwargs,
) -> xr.Dataset:
    """Read NetCDF, cast *_id coordinates to integers, sort coordinates.

    Extracted from: xarray_functions.py:46
    """

def write_netcdf(
    ds: xr.Dataset,
    filepath: str | Path,
    max_retries: int = 3,
    engine: str = 'netcdf4',
    compression: bool = True,
    compression_level: int = 4,
    chunking: bool = True,
    chunk_threshold: int = 1_000_000,
    max_chunk_size: int = 1000,
    manual_chunks: dict | None = None,
    chunk_by_dim: dict | None = None,
    use_temp_file: bool = True,
    mkdir: bool = True,                   # NEW: create parent directories (matches write_parquet)
    **kwargs,
) -> bool:
    """Write NetCDF with compression, chunking, atomic rename, chmod 0o775, and validation.

    Supersedes loading_functions.write_netcdf.
    Extracted from: xarray_functions.py:53
    Behavior change: mkdir parameter added; creates parent dirs when True (default).
    """

def convert_to_xarray(
    df: pd.DataFrame,
    dimensions: list[str] | None = None,
    dimension_dtypes: dict[str, str] | None = None,
    variable_dtypes: dict[str, str] | None = None,
    auto_optimize_dtypes: bool = True,
    validate_dimensions: bool = True,
) -> xr.Dataset:
    """Convert DataFrame to xarray Dataset with auto-detected or specified dimensions and dtypes.

    Extracted from: xarray_functions.py:353
    """

def convert_with_preset(
    df: pd.DataFrame,
    preset: str = 'as_variables',   # 'as_variables' | 'aa_variables'
    **kwargs,
) -> xr.Dataset:
    """Convert DataFrame to xarray using a named preset dimension configuration.

    Presets: 'as_variables' (location, year, age, sex), 'aa_variables' (location, year).
    Extracted from: xarray_functions.py:657
    """

def filter_ds_by_multiple_coords(
    ds: xr.Dataset,
    **coord_filters,   # coord_name=value_or_list
) -> xr.Dataset:
    """Filter xarray Dataset by one or more coordinate values.

    Extracted from: xarray_functions.py:673
    """

def filter_ds_by_range(
    ds: xr.Dataset,
    **coord_ranges,   # coord_name=(min_val, max_val)
) -> xr.Dataset:
    """Filter xarray Dataset by inclusive coordinate value ranges.

    Extracted from: xarray_functions.py:843
    """

def cast_coordinate_types(ds: xr.Dataset) -> xr.Dataset:
    """Cast standard ID coordinates to memory-efficient integer types.

    Extracted from: xarray_functions.py:33
    """

def sort_coordinates(
    ds: xr.Dataset,
    coords: list[str] | None = None,
    prioritize: list[str] | None = None,
) -> xr.Dataset:
    """Sort Dataset coordinates; location_id and year_id sort first by default.

    Extracted from: xarray_functions.py:236
    """
```

---

## lib/io/hdf5.py

Moves `hd5_functions.py` as-is. Single implementation; no consolidation needed.

```python
def write_hdf(
    df: pd.DataFrame,
    filepath: str | Path,
    key: str = 'df',
    max_retries: int = 3,
    validate: bool = True,
    compression: str = 'blosc:zstd',
    complevel: int = 9,
    **kwargs,
) -> bool:
    """Write DataFrame to HDF5 with exponential-backoff retry and full read-back validation.

    Creates parent directories. Sets chmod 0o775.
    Extracted from: hd5_functions.py:8
    """

def create_hdf_structure(
    file_path: str | Path,
    metadata_df: pd.DataFrame,
    draw_columns: list[str],
    metadata_columns: list[str],
) -> None:
    """Create HDF5 file with metadata datasets and pre-allocated draw columns.

    Extracted from: hd5_functions.py:189
    """

def write_draw_column(
    file_path: str | Path,
    draw_column: str,
    values: np.ndarray,
) -> None:
    """Write values to a single draw column in an existing HDF5 file.

    Extracted from: hd5_functions.py:225
    """

def read_hdf_metadata(
    file_path: str | Path,
    metadata_columns: list[str],
) -> pd.DataFrame:
    """Read only metadata columns from HDF5 file (avoids loading draw data).

    Extracted from: hd5_functions.py:244
    """
```

---

## lib/data/hierarchy.py

Centralizes hierarchy loading. `level_filter()` from `helper_functions.py` is canonical.

```python
def load_hierarchy(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load full LSAE hierarchy parquet.

    Defaults to constants.PROCESSED_DATA_PATH/full_hierarchy_lsae_1209.parquet.
    Extracted from: inline pattern in 12+ scripts (e.g., 00_make_covariate_means.py:30)
    """

def level_filter(
    hierarchy_df: pd.DataFrame,
    start_level: int,
    end_level: int | None = None,
    return_ids: bool = False,
) -> tuple | list[int]:
    """Return a parquet filter tuple (or location_id list) for a level range.

    Single level: level_filter(df, 5) → ('location_id', 'in', [...])
    Range: level_filter(df, 3, 5) → filter for levels 3 through 5.
    With return_ids=True: returns list[int] instead of tuple.
    This is the canonical implementation. Extracted from: helper_functions.py:92
    Note: most scripts inline this logic; Phase 4 replaces all with this call.
    """

def get_location_ids(
    hierarchy_df: pd.DataFrame,
    levels: int | list[int],
    extra_filter: pd.Series | None = None,
) -> list[int]:
    """Return location_ids for one or more levels, with optional boolean mask.

    extra_filter supports the as_malaria_fractions.py pattern of filtering on
    gbd_location_id membership before extracting location_ids.
    Extracted from: inline pattern in multiple scripts; extra_filter handles
    the special case in 04_forecasting/as_malaria_fractions.py:204
    """

def make_location_filter(location_ids: list[int]) -> tuple:
    """Return ('location_id', 'in', location_ids) parquet filter tuple.

    Covers L1 pattern. Extracted from: inline across all pipeline stages.
    """
```

---

## lib/data/covariates.py

Centralizes all model-predictor covariate loading. Covers H2 pattern plus
`read_income_paths`, `read_urban_paths`, and `merge_dataframes` from `helper_functions.py`.

```python
# Canonical clip bounds applied to all covariates regardless of disease.
# Disease-specific rules should be merged on top of these, not replace them.
UNIVERSAL_COVARIATE_CLIP_RULES: dict[str, tuple[float, float]] = {
    "relative_humidity": (0.001, 99.999),
    # Add future universal rules here
}

def load_covariates_for_draw(
    covariate_paths: dict[str, str],
    draw: str,
    ssp_scenario: str,
    climate_data_path: str | Path,
    filters: list | None = None,
    clip_rules: dict[str, tuple[float, float]] = UNIVERSAL_COVARIATE_CLIP_RULES,
    extra_clip_rules: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Load one draw-column per covariate file, rename to key, return merged DataFrame.

    clip_rules defaults to UNIVERSAL_COVARIATE_CLIP_RULES (applied to all diseases).
    extra_clip_rules are merged on top for disease-specific additions.
    Final applied rules = {**clip_rules, **(extra_clip_rules or {})}.

    Note: the RH clip was only present in dengue scripts historically, but is a universal
    covariate rule — malaria was missing it. Both callers should use the default.
    Extracted from: 02_data_prep/forecasted_draw_specific_malaria_dataframes.py:271
                    02_data_prep/forecasted_draw_specific_dengue_dataframes.py:118
    """

def read_income_paths(
    income_paths: dict[str, str],
    rcp_scenario: str,
    variable_data_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load income covariate files, filter to rcp_scenario, drop scenario column.

    Returns dict of DataFrames keyed by covariate name.
    Extracted from: helper_functions.py:67
    """

def read_urban_paths(
    urban_paths: dict[str, str],
    variable_data_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load urban covariate files with column renaming normalization.

    Normalizes: '300.0_simple_mean'→'300', '1500.0_simple_mean'→'1500',
    '100m_urban'→'urban_100m', '1km_urban'→'urban_1km', drops 'weighted_' prefix,
    drops 'population' column.
    Extracted from: helper_functions.py:77
    """

def merge_dataframes(
    model_df: pd.DataFrame,
    dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Left-merge a dict of DataFrames onto model_df on ['location_id', 'year_id'].

    Suffixes collisions with _{key}. Used after read_income_paths / read_urban_paths.
    Extracted from: helper_functions.py:62
    """
```

---

## lib/processing/raking.py

Preserves all three raking methods as separate functions. H4 finding: count-based
and logit-shift methods are not consolidatable.

```python
def rake_level(
    count_variable: str,
    level_df: pd.DataFrame,
    level_m1_df: pd.DataFrame,
    problematic_rules: dict,
    hierarchy_df: pd.DataFrame,
    level: int,
) -> pd.DataFrame:
    """Rake one hierarchy level to its parent level using count-based ratio raking.

    Uses population-based factor when count_raking_factor exceeds problematic_rules thresholds.
    Extracted from: rake_and_aggregate_functions.py:85
    """

def rake_aa_count_lsae_to_gbd(
    count_variable: str,
    hierarchy_df: pd.DataFrame,
    aa_gbd_count_df: pd.DataFrame,
    aa_lsae_count_df: pd.DataFrame,
    problematic_rules: dict,
    aa_full_count_df_path: str | Path | None = None,
    return_full_df: bool = False,
) -> pd.DataFrame:
    """Rake LSAE all-age counts to match GBD all-age counts at levels 4 and 5.

    Orchestrates rake_level() iteratively. Entry point for AA raking.
    Extracted from: rake_and_aggregate_functions.py:212
    """

def logit_shift_rake(
    forecast_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    rate_column: str,
    rake_year: int = 2022,
    clip_upper: float = 0.99,
) -> pd.DataFrame:
    """Rake by computing and applying a logit-space additive shift.

    shift = logit(observed_rate_at_rake_year) - logit(predicted_rate_at_rake_year).
    Applied additively to all forecast years. Preserves valid probability range.
    Used for dengue CFR raking only.
    Extracted from: 04_forecasting/rake_dengue.py:162-176
    """
```

---

## lib/processing/aggregation.py

Covers M1 pattern. AA and AS variants unified via `preserve_age_sex` parameter.

```python
def aggregate_level(
    count_variable: str,
    level_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate level_df counts to parent level by summing over parent_id and year_id.

    AA version: does not preserve age_group_id or sex_id.
    Extracted from: rake_and_aggregate_functions.py:284
    """

def aggregate_aa_count_lsae_to_gbd(
    count_variable: str,
    hierarchy_df: pd.DataFrame,
    aa_lsae_count_df: pd.DataFrame,
    aa_full_count_df_path: str | Path | None = None,
    return_full_df: bool = False,
) -> pd.DataFrame:
    """Aggregate LSAE all-age counts up to GBD hierarchy levels (4→0).

    Extracted from: rake_and_aggregate_functions.py:301
    """

def aggregate_aa_rate_lsae_to_gbd(
    rate_variable: str,
    hierarchy_df: pd.DataFrame,
    aa_lsae_rate_df: pd.DataFrame,
    aa_full_population_df: pd.DataFrame,
    aa_full_rate_df_path: str | Path | None = None,
    return_full_df: bool = False,
) -> pd.DataFrame:
    """Aggregate LSAE all-age rates by converting rate→count→aggregate→rate.

    Extracted from: rake_and_aggregate_functions.py:342
    """

def aggregate_to_parent(
    df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    count_variable: str,
    preserve_age_sex: bool = False,
) -> pd.DataFrame:
    """Generic aggregate-to-parent; wraps AA and AS logic via preserve_age_sex.

    preserve_age_sex=False: groups on [parent_id, year_id] (AA variant).
    preserve_age_sex=True:  groups on [parent_id, year_id, age_group_id, sex_id] (AS variant).
    Thin wrapper over aggregate_level() / cause_as_aggregation_by_draw logic.
    Extracted from: rake_and_aggregate_functions.py:284
                    05_aggregation/cause_as_aggregation_by_draw.py:76
    """
```

---

## lib/processing/disaggregation.py

Covers M2 pattern. Two separate functions — logic is structurally different.
Canonical malaria method: `as_malaria_fractions.py` (normalized RR fractions).
Decision recorded: 2026-03-27.

```python
def disaggregate_age_sex_malaria(
    aa_df: pd.DataFrame,            # all-age counts: location_id, year_id, aa_malaria_mort_count, aa_malaria_inc_count
    as_population_df: pd.DataFrame, # age-sex population: location_id, year_id, age_group_id, sex_id, population
    rr_df: pd.DataFrame,            # relative risks: gbd_location_id, age_group_id, sex_id, rr_inc_as, rr_mort_as
    hierarchy_df: pd.DataFrame,     # for location_id → gbd_location_id mapping
) -> pd.DataFrame:
    """Disaggregate all-age malaria counts to age-sex using normalized RR fractions.

    Method:
      rr_pop = rr * population
      fraction = rr_pop / sum(rr_pop)  [normalized so fractions sum to 1]
      count = fraction * aa_count
      age_group_id == 2 (early infancy) forced to zero.

    Canonical method. Extracted from: 04_forecasting/as_malaria_fractions.py:236-255
    """

def disaggregate_age_sex_dengue(
    as_population_df: pd.DataFrame,   # age-sex population: location_id, year_id, age_group_id, sex_id, population
    aa_forecast_df: pd.DataFrame,     # all-age forecasts: location_id, year_id, base_log_dengue_inc_rate_pred, dengue_cfr_pred
    rr_df: pd.DataFrame,              # relative risks: gbd_location_id, age_group_id, sex_id, rr_inc_as
    hierarchy_df: pd.DataFrame,       # for location_id → gbd_location_id mapping
) -> pd.DataFrame:
    """Disaggregate all-age dengue forecasts to age-sex using log-rate + CFR method.

    Method:
      inc_count = population * exp(base_log_inc_rate) * rr_inc
      mort_count = inc_count * cfr

    Extracted from: 04_forecasting/as_dengue_shifts.py:202-203
    """
```

---

## lib/processing/scenarios.py

Covers M3 pattern. Malaria-only for now.

```python
def generate_dah_scenarios(
    baseline_df: pd.DataFrame,
    ssp_scenario: str,
    year_start: int = 2000,
    reference_year: int = 2023,
    modification_start_year: int = 2026,
    dah_scenario_names: list[str] | None = None,
) -> tuple[list[str], list[pd.DataFrame]]:
    """Generate four DAH funding scenarios for malaria forecasting.

    Scenarios:
      Baseline: original projections unchanged.
      Constant:  DAH frozen at reference_year levels from 2024 onward.
      Increasing: multipliers 1.2→1.4→1.6→1.8→2.0 applied 2026–2030+.
      Decreasing: multipliers 0.8→0.6→0.4→0.2→0.0 applied 2026–2030+.

    Returns (scenario_names, scenario_dataframes).
    Extracted from: 02_data_prep/forecasted_draw_specific_malaria_dataframes.py:76
    """
```

---

## lib/utils/transforms.py

Small math helpers used inside lib functions, not directly by pipeline scripts.

```python
def logit(p: np.ndarray | pd.Series, clip_upper: float = 0.99) -> np.ndarray | pd.Series:
    """Logit transform: log(p / (1 - p)). Clips at clip_upper to avoid inf.

    Extracted from: inline in 04_forecasting/rake_dengue.py:166
    """

def expit(x: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Inverse logit: 1 / (1 + exp(-x)). Maps real line to (0, 1).

    Used when applying logit-shift raking results back to probability space.
    """
```

---

## Decisions encoded in this design

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | `write_parquet(use_atomic=True)` default | Matches `write_netcdf` behavior; prevents partial writes |
| D2 | `write_netcdf` gains `mkdir=True` parameter | Matches `write_parquet` behavior; callers should not need to pre-create dirs |
| D3 | `level_filter()` from `helper_functions.py` is canonical | User confirmed 2026-03-27 |
| D4 | `disaggregate_age_sex_malaria` uses normalized RR fractions | `as_malaria_fractions.py` is canonical; `as_malaria_shifts.py` is superseded |
| D5 | RH clip `[0.001, 99.999]` is a universal covariate rule | Malaria was missing it historically; both diseases apply it via `UNIVERSAL_COVARIATE_CLIP_RULES` default |
| D6 | `data/covariates.py` (not `climate.py`) | Covers all model-predictor loading (climate + income + urban) |
| D7 | Raking has three separate functions | Count-based ratio, logit-shift, and spatial methods are not consolidatable |

---

## What is NOT in lib/

| Item | Reason |
|------|--------|
| `loading_functions.write_parquet` | Superseded by `lib/io/parquet.write_parquet` |
| `loading_functions.write_netcdf` | Superseded by `lib/io/netcdf.write_netcdf` |
| `as_malaria_shifts.py` disaggregation | Superseded by `disaggregate_age_sex_malaria` (fractions method) |
| `data/population.py`, `data/gbd.py` | No duplicated loading pattern found in audit |
| `utils/validation.py` | Pipeline principle: no defensive coding for internal data |
| Spatial raking (`sum_and_align_admin2_totals`, `broadcast_factor_to_admin2`) | Complex; tied to `05_aggregation/create_raked_outcomes.py` structure. Extract in Phase 3 after reviewing that script. |
