# Session memory
Updated: 2026-05-07

## Current task
Script 05 (malaria past inputs netCDF) is complete and working. Dengue (06) is deferred.

## Context / why
Replacing ~5TB parquet-per-draw storage with dimension-aware xarray/netCDF. Past inputs carry only the dimensions variables actually vary over.

## Where we are
`05_build_malaria_past_inputs.py` runs clean and produces `malaria_past_inputs.nc` (0.09 GB).
Output at: `/mnt/team/idd/pub/forecast-mbp/03-modeling_data/malaria/past_inputs_nc/lsae_1285/20260405/malaria_past_inputs.nc`

Variables in the file:
- `loc×year`: population, malaria_pfpr, set_by_gbd, malaria_inc/mort rate/count, mal_DAH_total, mal_DAH_total_per_capita, gdppc_mean, ldipc_mean, 4 urban vars, people_flood_days, people_flood_days_per_capita
- `loc×year×draw`: total_precipitation, precipitation_days, relative_humidity, wind_speed, mean_temperature, mean_low_temperature, mean_high_temperature, days_over_30C
- `loc×year×draw×suit_variant`: malaria_suitability (currently 1 variant since all 14 point to same file)

## Next steps
1. Move on to whatever comes after past inputs in the malaria pipeline (R model fitting or future inputs)
2. When returning to dengue: apply same fixes to `06_build_dengue_past_inputs.py` (see below)

## Dengue 06 — fixes needed when we return
All fixes applied to 05 need to be mirrored in 06:
- `wide_to_array` fix is shared (already fixed in `lib/io/array_builders.py`) ✓
- `rcp_scenario` filter for gdppc/ldipc: already added to 06 ✓
- `dengue_suitability` via extra_vars: already added to 06 ✓
- `add_base` flag: NOT added to 06 yet (dengue R model fits on AS data directly, so base_* may or may not be relevant — decide when returning to dengue)

## Resume prompt
Script 05_build_malaria_past_inputs.py is complete and working. It produces malaria_past_inputs.nc (0.09 GB) with AA outcomes + covariates (loc×year), 8 climate vars (loc×year×draw), and malaria_suitability (loc×year×draw×suit_variant). Dengue script 06 is deferred. The shared fix to wide_to_array (climate parquets store loc/year as MultiIndex, not columns — fix: read columns=DRAWS only then reset_index()) is already in lib/io/array_builders.py. Next step is whatever comes after past inputs in the malaria pipeline.
