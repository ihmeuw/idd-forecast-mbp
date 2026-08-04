"""Helpers for building numpy arrays from wide-format climate parquets.

Used by build_malaria_past_inputs.py and build_dengue_past_inputs.py.
"""
import numpy as np
import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids

DRAWS = mbpc.draws  # ['000', '001', ..., '099']


def wide_to_array(
    path: str,
    location_ids: list[int],
    years: list[int],
) -> np.ndarray:
    """Read wide-format climate parquet → float32 array of shape (n_loc, n_year, n_draw).

    Handles parquets where location_id/year_id are stored as the MultiIndex
    (draw columns '000'..'099' are the only data columns).
    Missing location×year combinations are filled with NaN.
    """
    df = pd.read_parquet(
        str(path),
        columns=DRAWS,
        filters=[('location_id', 'in', location_ids), ('year_id', 'in', years)],
    ).reset_index()
    full_idx = pd.MultiIndex.from_product(
        [location_ids, years], names=['location_id', 'year_id']
    )
    return (
        df.set_index(['location_id', 'year_id'])[DRAWS]
        .reindex(full_idx)
        .values
        .reshape(len(location_ids), len(years), len(DRAWS))
        .astype(np.float32)
    )


def scalar_to_array(
    df: pd.DataFrame,
    col: str,
    location_ids: list[int],
    years: list[int],
) -> np.ndarray:
    """Pivot a location×year DataFrame column → float32 array of shape (n_loc, n_year).

    Missing location×year combinations are filled with NaN.
    """
    full_idx = pd.MultiIndex.from_product(
        [location_ids, years], names=['location_id', 'year_id']
    )
    return (
        df.set_index(['location_id', 'year_id'])[[col]]
        .reindex(full_idx)[col]
        .values
        .reshape(len(location_ids), len(years))
        .astype(np.float32)
    )


SHARED_COVARIATE_SOURCES: dict[str, set[str]] = {
    # Source key -> set of variable names that source contributes to `arrays`.
    # A source is loaded iff at least one of its variables is in the request.
    "gdppc":         {"gdppc_mean"},
    "ldipc":         {"ldipc_mean"},
    "urban_300":     {"weighted_1km_urban_threshold_300.0_simple_mean"},
    "urban_1500":    {"weighted_1km_urban_threshold_1500.0_simple_mean"},
    "flooding":      {"people_flood_days_per_capita"},
    "med_consumppc": {"med_consumppc"},
}
ALL_SHARED_VARIABLES: frozenset[str] = frozenset().union(*SHARED_COVARIATE_SOURCES.values())


def read_shared_covariates(
    location_ids: list[int],
    years: list[int],
    gdppc_read_path=None,
    ldipc_read_path=None,
    urban_read_path=None,
    flooding_path: str | None = None,
    rcp_scenario: str = "rcp45",
    med_consumppc_read_path=None,
    *,
    variables: "list[str] | tuple[str, ...] | set[str] | None" = None,
) -> dict:
    """Read non-draw scalar covariates shared by malaria and dengue past inputs.

    Returns a dict mapping variable name → (n_loc, n_year) float32 array.

    Parameters
    ----------
    variables:
        Iterable of variable names to load. Each must appear in
        `ALL_SHARED_VARIABLES`. Only the sources whose variables are requested
        are read from disk — useful for callers that need a subset (e.g. 08
        passing through its --covariates flag, 07b checking only a few vars).
        If None (default), every source with a non-None path is loaded
        (backward-compat with pre-refactor callers).
    rcp_scenario:
        Filter for gdppc/ldipc/med_consumppc (which store all scenarios). Pass
        the string label ("rcp26", "rcp45", "rcp85"). Default "rcp45" (SSP245).
    flooding_path:
        Direct path to the flooding parquet. Warns and omits if None / missing.
    """
    import warnings
    from pathlib import Path

    if variables is None:
        requested_sources = set(SHARED_COVARIATE_SOURCES)  # legacy: load all
    else:
        requested = set(variables)
        unknown = requested - ALL_SHARED_VARIABLES
        if unknown:
            raise ValueError(
                f"Unknown shared variable(s): {sorted(unknown)}. "
                f"Known: {sorted(ALL_SHARED_VARIABLES)}"
            )
        requested_sources = {
            src for src, vars_ in SHARED_COVARIATE_SOURCES.items() if vars_ & requested
        }

    loc_filter = ('location_id', 'in', location_ids)
    year_filter = ('year_id', 'in', years)
    arrays = {}

    # GDP per capita
    if "gdppc" in requested_sources and gdppc_read_path is not None:
        gdppc_df = read_parquet_with_integer_ids(
            str(Path(gdppc_read_path) / "gdppc_mean.parquet"),
            filters=[loc_filter, year_filter, ('scenario', '==', rcp_scenario)],
        )
        gdppc_df = gdppc_df.drop(columns=['scenario'], errors='ignore')
        for col in [c for c in gdppc_df.columns if c not in ('location_id', 'year_id')]:
            arrays[col] = scalar_to_array(gdppc_df, col, location_ids, years)

    # LDIPC
    if "ldipc" in requested_sources and ldipc_read_path is not None:
        ldipc_df = read_parquet_with_integer_ids(
            str(Path(ldipc_read_path) / "ldipc_mean.parquet"),
            filters=[loc_filter, year_filter, ('scenario', '==', rcp_scenario)],
        )
        ldipc_df = ldipc_df.drop(columns=['scenario'], errors='ignore')
        for col in [c for c in ldipc_df.columns if c not in ('location_id', 'year_id')]:
            arrays[col] = scalar_to_array(ldipc_df, col, location_ids, years)

    # Urban thresholds — separate parquets per threshold
    urban_files = {
        "urban_300":  ("urban_threshold_300.0_simple_mean.parquet"),
        "urban_1500": ("urban_threshold_1500.0_simple_mean.parquet"),
    }
    if any(src in requested_sources for src in urban_files) and urban_read_path is not None:
        for src, fname in urban_files.items():
            if src not in requested_sources:
                continue
            udf = read_parquet_with_integer_ids(
                str(Path(urban_read_path) / fname),
                filters=[loc_filter, year_filter],
            )
            for col in [c for c in udf.columns if c not in ('location_id', 'year_id', 'population')]:
                arrays[col] = scalar_to_array(udf, col, location_ids, years)

    # Flooding
    if "flooding" in requested_sources:
        fpath = Path(flooding_path) if flooding_path else None
        if fpath and fpath.exists():
            fdf = read_parquet_with_integer_ids(
                str(fpath), filters=[loc_filter, year_filter]
            )
            fdf = fdf.drop(columns=['model', 'scenario', 'variant', 'population'], errors='ignore')
            for col in [c for c in fdf.columns if c not in ('location_id', 'year_id')]:
                arrays[col] = scalar_to_array(fdf, col, location_ids, years)
        else:
            warnings.warn(f"Flooding data not found at {flooding_path}; omitted from past inputs.")

    # Median consumption per capita
    if "med_consumppc" in requested_sources and med_consumppc_read_path is not None:
        mcp_df = read_parquet_with_integer_ids(
            str(Path(med_consumppc_read_path) / "med_consumppc_mean.parquet"),
            filters=[loc_filter, year_filter, ('scenario', '==', rcp_scenario)],
        )
        mcp_df = mcp_df.drop(columns=['scenario'], errors='ignore')
        for col in [c for c in mcp_df.columns if c not in ('location_id', 'year_id')]:
            arrays[col] = scalar_to_array(mcp_df, col, location_ids, years)

    return arrays


def read_draw_climate(
    location_ids: list[int],
    years: list[int],
    ssp_scenario: str,
    lsae_hierarchy: str,
    extra_vars: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Read all draw-varying climate variables → dict of (n_loc, n_year, n_draw) arrays.

    extra_vars: additional {var_name: path} entries beyond the standard set.
    """
    CLIMATE = mbpc.CLIMATE_AGGREGATES_PATH / lsae_hierarchy
    climate_vars = {
        'total_precipitation':   str(CLIMATE / f"total_precipitation_{ssp_scenario}.parquet"),
        'precipitation_days':    str(CLIMATE / f"precipitation_days_{ssp_scenario}.parquet"),
        'relative_humidity':     str(CLIMATE / f"relative_humidity_{ssp_scenario}.parquet"),
        'wind_speed':            str(CLIMATE / f"wind_speed_{ssp_scenario}.parquet"),
        'mean_temperature':      str(CLIMATE / f"mean_temperature_{ssp_scenario}.parquet"),
        'mean_low_temperature':  str(CLIMATE / f"mean_low_temperature_{ssp_scenario}.parquet"),
        'mean_high_temperature': str(CLIMATE / f"mean_high_temperature_{ssp_scenario}.parquet"),
        'days_over_30C':         str(CLIMATE / f"days_over_30C_{ssp_scenario}.parquet"),
    }
    if extra_vars:
        climate_vars.update(extra_vars)

    arrays = {}
    for var_name, path in climate_vars.items():
        print(f"  {var_name}...")
        arrays[var_name] = wide_to_array(path, location_ids, years)
    return arrays
