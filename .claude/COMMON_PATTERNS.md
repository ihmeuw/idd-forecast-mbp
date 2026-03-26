# Common Patterns
Generated: 2026-03-26
Phase 1, Task 1.3

Patterns are categorized by frequency:
- **HIGH** (≥5 occurrences): Review in detail before extracting
- **MEDIUM** (3–4 occurrences): Quick approval unless something looks wrong
- **LOW** (1–2 occurrences): Can wait or bundle

---

## HIGH Priority Patterns

---

### H1: Hierarchy Loading + Level Filtering
**Frequency:** 12+ locations across all stages
**Priority:** HIGH

#### What it does
Loads the full LSAE hierarchy parquet, extracts location_ids at a specific level, and creates a parquet filter tuple for downstream reads.

#### Canonical form (from `00_make_covariate_means.py:30–35` and `07_forecasted_dataframes_non_draw_part.py:32–36`)
```python
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)
md_location_ids = hierarchy_df[hierarchy_df['level'] == 5]['location_id'].unique().tolist()
md_location_filter = ('location_id', 'in', md_location_ids)
```

#### All occurrences
| File | Lines | Level(s) | Notes |
|------|-------|----------|-------|
| `02_data_prep/00_make_covariate_means.py` | 30–35 | 5 | md_location_filter |
| `02_data_prep/07_forecasted_dataframes_non_draw_part.py` | 32–36 | 5 | md_location_filter |
| `02_data_prep/01_make_full_hierarchy.py` | 69, 87, 100, 301 | 1, 3, 4, 5 | Multiple separate filters |
| `02_data_prep/02_as_fhs_and_full_population.py` | 327, 346, 374, 403–404, 462 | 3, 4, 5, <=3, >=3 | Inline throughout |
| `04_forecasting/as_malaria_fractions.py` | 144, 204–205 | 5 + gbd_location_id condition | Extra join condition |
| `rake_and_aggregate_functions.py` | 217, 238–244, 304–309 | 3, 4, 5 | Direct column filter |
| `05_aggregation/cause_as_aggregation_by_draw.py` | 71–125 | 1–5 (loop) | Hierarchical iteration |
| `05_aggregation/cause_as_aggregation_by_draw_raked.py` | 111–173 | 1–5 (loop) | Same, with raking |
| `06_upload/create_and_combine_aa_draws.py` | 111–128 | All | No level filter |
| `06_upload/combine_as_draws.py` | 99–118 | FHS membership | Different filter logic |
| `01_map_to_admin_2/pixel_hierarchy.py` | 106–130 | None | Loads subset, no filter |
| `01_map_to_admin_2/pixel_urban_hierarchy.py` | 104–128 | None | Same as pixel_hierarchy |

#### Existing helper
`level_filter()` in `helper_functions.py:92–107` — returns a filter tuple, but is NOT consistently used. Most scripts inline the same logic directly.

#### Implementation divergences
1. **Most common** (level == N): `hierarchy_df[hierarchy_df['level'] == 5]['location_id'].unique().tolist()` — returns list, builds filter tuple separately
2. **Range filter** (level >= N): `hierarchy_df[hierarchy_df['level'] >= 3]` — used in population prep
3. **With extra condition**: `hierarchy_df[(hierarchy_df['level'] == 5) & (hierarchy_df['gbd_location_id'].isin(...))]` — as_malaria_fractions only
4. **Iterative**: `for level in reversed(range(1, 6))` — aggregation scripts

#### Proposed shared functions
```python
# lib/data/hierarchy.py
def load_hierarchy(path: str | Path = None) -> pd.DataFrame:
    """Load full LSAE hierarchy. Defaults to PROCESSED_DATA_PATH/full_hierarchy_lsae_1209.parquet."""

def get_location_ids(hierarchy_df: pd.DataFrame, levels: int | list[int]) -> list[int]:
    """Return location_ids for given level(s)."""

def make_location_filter(location_ids: list[int]) -> tuple:
    """Return ('location_id', 'in', location_ids) filter tuple for parquet reads."""
```

#### ⚠️ Decision needed
- Is `level_filter()` in `helper_functions.py` the intended canonical version, or did it get bypassed accidentally?
- `as_malaria_fractions.py` adds a `gbd_location_id` join — is that a special case or should it be the default?

---

### H2: Climate/Covariate Loading by Draw
**Frequency:** 5 locations (malaria × 2 active versions + dengue × 1 + covariate_functions + data_functions)
**Priority:** HIGH

#### What it does
Iterates over a dict of covariate path templates, reads one draw column per file, renames it to the covariate key, and merges onto the forecast dataframe.

#### Canonical form (from `forecasted_draw_specific_malaria_dataframes.py:271–280`)
```python
for key, path_template in cc_sensitive_paths.items():
    path = path_template.format(CLIMATE_DATA_PATH=CLIMATE_DATA_PATH, ssp_scenario=ssp_scenario)
    columns_to_read = ["location_id", "year_id", draw]
    df = read_parquet_with_integer_ids(path, columns=columns_to_read, filters=[aa_malaria_filter])
    df = df.rename(columns={draw: key})
    forecast_by_draw_df = pd.merge(forecast_by_draw_df, df, on=["location_id", "year_id"], how="left")
```

#### All occurrences
| File | Lines | Draw usage | Covariates | Notes |
|------|-------|-----------|------------|-------|
| `02_data_prep/forecasted_draw_specific_malaria_dataframes.py` | 271–280 | draw as column name | total_precipitation, relative_humidity, malaria_suitability | Active |
| `02_data_prep/forecasted_draw_specific_dengue_dataframes.py` | 118–126 | draw as column name | total_precipitation, relative_humidity, dengue_suitability | Active; clips rh to [0.001, 99.999] |
| `02_data_prep/forecasted_draw_specific_malaria_dataframes_old.py` | 249–260 | draw as column name | total_precipitation, malaria_suitability | Old; skips flooding key |
| `covariate_functions.py` | 30–54 | ALL draws at once (wide format) | malaria/dengue_suitability | Used for visualization; converts to xarray |
| `data_functions.py` | 47–49, 141–174 | draw in file PATH | suitability raster .nc | Used for raster viz; hardcoded to draw=000 |

#### Implementation divergences
1. **Per-draw column selection** (pipeline scripts): `columns_to_read = ["location_id", "year_id", draw]` — most common
2. **All-draws wide** (`covariate_functions.py`): reads all 100 draw columns at once, melts to long format for xarray
3. **Draw in path** (`data_functions.py`): separate .nc file per draw; currently hardcoded to draw=000

#### The dengue clip rule
Dengue clips relative_humidity before merging: `df['relative_humidity'] = df['relative_humidity'].clip(lower=0.001, upper=99.999)`. Malaria does not. This is a disease-specific rule.

#### Proposed shared function
```python
# lib/data/climate.py
def load_covariates_for_draw(
    covariate_paths: dict[str, str],
    draw: str,
    ssp_scenario: str,
    climate_data_path: str | Path,
    filters: list = None,
    clip_rules: dict[str, tuple[float, float]] = None,  # e.g. {"relative_humidity": (0.001, 99.999)}
) -> pd.DataFrame:
    """Load one draw-column per covariate file, rename, and return merged DataFrame."""
```

#### ⚠️ Decision needed
- The relative_humidity clip in dengue — intentional disease-specific rule, or should malaria also clip?
- `covariate_functions.py` loads all draws at once for visualization. Should this be a separate function signature, or a parameter (`load_all_draws=True`)?

---

### H3: File Write with Retry and Permissions
**Frequency:** Used across all 6 stages (10+ call sites); 4 wrapper implementations exist
**Priority:** HIGH

#### What it does
Write parquet/netCDF/HDF5 with retry logic, atomic temp-file rename, directory creation, and chmod 0o775.

#### Current state: duplication problem
There are two sets of write functions. The `loading_functions.py` versions duplicate `parquet_functions.py` and `xarray_functions.py` with simpler/older implementations.

| Function | Module | Retries | Sleep | Atomic | mkdir | chmod | Validation |
|----------|--------|---------|-------|--------|-------|-------|------------|
| `write_parquet()` | `parquet_functions.py` | 3 | No | Optional | Yes | Yes | metadata/sample/full |
| `write_parquet()` | `loading_functions.py` | 3 | No | Always | No | No | full readback |
| `write_netcdf()` | `xarray_functions.py` | 3 | No | Always | No | Yes | metadata |
| `write_netcdf()` | `loading_functions.py` | 3 | No | Always | No | No | full readback |
| `write_hdf()` | `hd5_functions.py` | 3 | Exp backoff | No | Yes | Yes | full readback |

#### Additional gap: direct writes without wrappers
- `01_map_to_admin_2/01_prep_maps.py:203, 277, 404` — direct `ds.to_netcdf()` (has custom `set_file_permissions()` helper but no retry)
- `02_data_prep/01_make_full_hierarchy.py` — direct `.to_parquet()` for lookup tables

#### Proposed lib structure
```python
# lib/io/parquet.py  — consolidate parquet_functions.py (canonical) + remove loading_functions.write_parquet
# lib/io/netcdf.py   — consolidate xarray_functions.py (canonical) + remove loading_functions.write_netcdf
# lib/io/hdf5.py     — hd5_functions.py as-is (only one version)
```

#### ⚠️ Decision needed
- `xarray_functions.write_netcdf()` does NOT create parent dirs; `parquet_functions.write_parquet()` does. Should netCDF also mkdir?
- The `parquet_functions.write_parquet()` `use_atomic` parameter defaults to `False`. Should it default to `True` to match the netCDF behavior?

---

### H4: Raking (Subnational → National/GBD)
**Frequency:** 5 distinct implementations across 4 files
**Priority:** HIGH

#### What it does
Reconciles subnational estimates to national/GBD totals by computing and applying raking factors.

#### All occurrences
| File | Lines | Method | Disease | Notes |
|------|-------|--------|---------|-------|
| `rake_and_aggregate_functions.py:85–210` | `rake_level()` | Count-based ratio | Both | Core function; handles problematic_rules thresholds |
| `rake_and_aggregate_functions.py:212–262` | `rake_aa_count_lsae_to_gbd()` | Orchestrates rake_level for L4, L5 | Both | Calls rake_level iteratively |
| `02_data_prep/03_rake_aa_A2_to_GBD.py` | Calls above | Wraps rake_aa_count_lsae_to_gbd | Both | Entry point for AA raking |
| `04_forecasting/rake_dengue.py:168–176` | Logit shift method | Dengue CFR only | Dengue | Different method: `shift = logit_cfr_obs - logit_cfr_pred`; adds shift forward in time |
| `05_aggregation/create_raked_outcomes.py:276–333` | Spatial: admin-2 → FHS admin-1 | Both | Uses `sum_and_align_admin2_totals()` then `broadcast_factor_to_admin2()` |

#### Key divergence: two fundamentally different raking methods
1. **Count-based ratio raking** (`rake_level`): `raking_factor = parent_count / sum_of_child_counts`; applied multiplicatively. Conditional logic in `problematic_rules` switches between count-based and population-based factors.
2. **Logit shift raking** (`rake_dengue.py`): `shift = logit(observed_2022_CFR) - logit(predicted_2022_CFR)`; applied additively in logit space. Preserves valid probability range.

These are not consolidatable into one function — they solve different problems.

#### Proposed lib structure
```python
# lib/processing/raking.py
# Keep rake_level() and rake_aa_count_lsae_to_gbd() from rake_and_aggregate_functions.py
# Keep logit shift raking from rake_dengue.py as separate function
# Keep spatial raking from create_raked_outcomes.py as separate function
```

---

## MEDIUM Priority Patterns

---

### M1: Hierarchy Aggregation (Level 5 → Level 4 → Level 3 → ...)
**Frequency:** 4 files
**Priority:** MEDIUM

#### What it does
Rolls up estimates from most-detailed (level 5) to higher geographic levels by summing counts over parent_id.

#### All occurrences
| File | Lines | Preserves age/sex | Weighting | Notes |
|------|-------|-------------------|-----------|-------|
| `rake_and_aggregate_functions.py:284–299` | `aggregate_level()` | No | None (sum) | AA version; collapses all dims except location, year |
| `rake_and_aggregate_functions.py:301–323` | `aggregate_aa_count_lsae_to_gbd()` | No | None | Iterates levels 4→0 |
| `rake_and_aggregate_functions.py:342–359` | `aggregate_aa_rate_lsae_to_gbd()` | No | Population | Converts rate→count→aggregate→rate |
| `05_aggregation/cause_as_aggregation_by_draw.py:76–126` | `process_forecast_data()` | **Yes** | None (sum) | AS version; groups on age_group_id, sex_id too |
| `05_aggregation/cause_as_aggregation_by_draw_raked.py:156–164` | Same + raking pre-applied | **Yes** | None | Identical to above except raking applied first |

#### Key divergence
- `rake_and_aggregate_functions.aggregate_level()` groups on `['parent_id', 'year_id']` — collapses age/sex
- `cause_as_aggregation_by_draw.process_forecast_data()` groups on `['parent_id', 'year_id', 'age_group_id', 'sex_id']` — preserves age/sex

These are the AA and AS variants of the same concept.

#### Proposed shared function
```python
# lib/processing/aggregation.py
def aggregate_to_parent(
    df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    count_variable: str,
    preserve_age_sex: bool = False,
) -> pd.DataFrame:
```

---

### M2: Age/Sex Disaggregation (All-age → Age-specific)
**Frequency:** 3 active files (2 malaria methods + 1 dengue)
**Priority:** MEDIUM

#### What it does
Disaggregates all-age estimates to age-sex specific using relative risk fractions from GBD.

#### All occurrences
| File | Lines | Method | Disease | Normalizes? |
|------|-------|--------|---------|-------------|
| `04_forecasting/as_malaria_fractions.py:236–252` | Normalized RR fractions | Malaria | **Yes** — divides by pop-weighted RR sum |
| `04_forecasting/as_malaria_shifts.py:195–196` | Direct RR multiplication | Malaria | No |
| `04_forecasting/as_dengue_shifts.py:202–203` | Direct RR × pop × exp(log_rate); mortality = incidence × CFR | Dengue | No |

#### Key divergences
1. **Malaria has two methods**:
   - `as_malaria_fractions.py`: `fraction = (rr × pop) / sum(rr × pop)`; then `count = fraction × aa_count`. Normalizes so fractions sum to 1. Also zeroes out age_group_id 2 (early infancy).
   - `as_malaria_shifts.py`: `count = population × base_rate × rr`. Simpler; no normalization.
2. **Dengue**: `count_inc = population × exp(base_log_inc_rate) × rr_inc`; `count_mort = count_inc × cfr`. Uses log-scale base rate; mortality derived from incidence via CFR.

#### ⚠️ Decision needed (HIGH importance)
Malaria has two disaggregation implementations. Which is canonical? `as_malaria_fractions.py` or `as_malaria_shifts.py`? They will produce different results. This must be resolved before Phase 3.

---

### M3: DAH Scenario Generation
**Frequency:** 3 malaria files (1 canonical function + 2 alternative scenario scripts)
**Priority:** MEDIUM

#### What it does
Creates 4 DAH scenario variants (Baseline, Constant, Increasing, Decreasing) by modifying `mal_DAH_total_per_capita` starting in 2026.

#### Canonical function
`generate_dah_scenarios()` in `02_data_prep/forecasted_draw_specific_malaria_dataframes.py:76–218`

**Scenario logic:**
- **Baseline**: Unchanged original projections
- **Constant**: DAH frozen at 2023 levels from 2024 onward
- **Increasing**: Multipliers 1.2→1.4→1.6→1.8→2.0 applied 2026–2030+
- **Decreasing**: Multipliers 0.8→0.6→0.4→0.2→0.0 applied 2026–2030+

#### All occurrences
| File | Lines | Notes |
|------|-------|-------|
| `02_data_prep/forecasted_draw_specific_malaria_dataframes.py:76–218` | Canonical implementation | Active |
| `02_data_prep/forecasted_draw_specific_malaria_dataframes_old.py:70–236` | Legacy duplicate | Old/archived |
| `02_data_prep/alt_forecasted_malaria_dataframes.py:52–134` | Alternative scenarios (GK_reference, GK_cut20) | Reads from external DAH files; no generation |
| `02_data_prep/BG_forecasted_malaria_dataframes.py:52–134` | Identical to alt_ script | Same |

#### Notes
- **Malaria only** — dengue has no DAH scenarios currently
- `alt_` and `BG_` scripts process externally-supplied DAH data (not generated); the 4-scenario logic is not duplicated there
- `05_aggregation` Jobmon scripts use `dah_scenarios = ['GK_reference_2025_11_02', 'GK_cut20_2025_11_02']` — these are the alt scenarios, not the standard 4

#### Proposed lib location
```python
# lib/processing/scenarios.py
def generate_dah_scenarios(baseline_df, ssp_scenario, ...) -> tuple[list, list]:
    """Extracted from forecasted_draw_specific_malaria_dataframes.py"""
```

---

## LOW Priority Patterns

---

### L1: Parquet Filter Tuple Construction
**Frequency:** Many locations, but trivial (1 line each)
**Priority:** LOW

The pattern `('location_id', 'in', location_ids)` appears everywhere parquet is filtered by location. Already partly captured by `level_filter()` in `helper_functions.py`. Not worth a dedicated function; fold into H1 hierarchy helpers.

---

### L2: Wide-to-Long Draw Conversion
**Frequency:** 2 locations (`covariate_functions.py`, potentially others)
**Priority:** LOW

`convert_draws_to_xarray()` in `covariate_functions.py:18–27` melts wide draw columns to long format, then converts to xarray. Used primarily for visualization. Keep as-is; fold into `lib/io/netcdf.py` or `lib/data/climate.py` during Phase 3.

---

## Summary Table

| Pattern | ID | Frequency | Priority | Proposed lib location | Decision needed? |
|---------|-----|-----------|----------|-----------------------|------------------|
| Hierarchy loading + level filter | H1 | 12+ | HIGH | `lib/data/hierarchy.py` | Which form is canonical? |
| Climate/covariate loading by draw | H2 | 5 | HIGH | `lib/data/climate.py` | RH clip rule; all-draws variant |
| File write with retry/permissions | H3 | 10+ | HIGH | `lib/io/` (3 files) | mkdir in netCDF? atomic default? |
| Raking | H4 | 5 | HIGH | `lib/processing/raking.py` | No — two methods are distinct |
| Hierarchy aggregation | M1 | 4 | MEDIUM | `lib/processing/aggregation.py` | AA vs AS is clear |
| Age/sex disaggregation | M2 | 3 | MEDIUM | `lib/processing/disaggregation.py` | **Which malaria method is canonical?** |
| DAH scenario generation | M3 | 3 | MEDIUM | `lib/processing/scenarios.py` | No |
| Parquet filter tuple | L1 | Many | LOW | Fold into H1 | No |
| Wide-to-long draw conversion | L2 | 2 | LOW | `lib/io/` or `lib/data/` | No |

---

## Open Questions for Bobby

Before Phase 2 (design), these must be resolved:

1. **[M2 — CRITICAL]** Malaria age/sex disaggregation: `as_malaria_fractions.py` (normalized RR fractions) vs `as_malaria_shifts.py` (direct multiplication). Which is the current/canonical production method? They produce different numbers.

2. **[H1]** The `level_filter()` function in `helper_functions.py` exists but most scripts inline the same logic. Was it always ignored, or did scripts diverge from it over time?

3. **[H2]** Does the relative_humidity clip (`[0.001, 99.999]`) in dengue apply to the modeling step only, or should it also be applied in malaria forecasts?

4. **[H3]** `write_netcdf()` does not create parent directories (relies on caller). `write_parquet()` does. Should we standardize to always mkdir in the write function?

5. **[H3]** `write_parquet(use_atomic=False)` by default. `write_netcdf()` always uses atomic write. Should parquet also default to atomic?
