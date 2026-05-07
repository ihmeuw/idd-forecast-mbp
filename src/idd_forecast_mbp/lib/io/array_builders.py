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


def read_shared_covariates(
    location_ids: list[int],
    years: list[int],
    gdppc_read_path,
    ldipc_read_path,
    urban_read_path,
    flooding_path: str | None,
    rcp_scenario: float = 4.5,
) -> dict:
    """Read non-draw scalar covariates shared by malaria and dengue past inputs.

    Returns a dict mapping variable name → (n_loc, n_year) float32 array.
    Flooding is optional; warns and omits if path is None or does not exist.
    rcp_scenario filters gdppc/ldipc which contain all scenarios; default 4.5 (SSP245).
    """
    import warnings
    from pathlib import Path

    loc_filter = ('location_id', 'in', location_ids)
    year_filter = ('year_id', 'in', years)
    arrays = {}

    # GDP per capita
    gdppc_df = read_parquet_with_integer_ids(
        str(Path(gdppc_read_path) / "gdppc_mean.parquet"),
        filters=[loc_filter, year_filter, ('scenario', '==', rcp_scenario)],
    )
    gdppc_df = gdppc_df.drop(columns=['scenario'], errors='ignore')
    for col in [c for c in gdppc_df.columns if c not in ('location_id', 'year_id')]:
        arrays[col] = scalar_to_array(gdppc_df, col, location_ids, years)

    # LDIPC
    ldipc_df = read_parquet_with_integer_ids(
        str(Path(ldipc_read_path) / "ldipc_mean.parquet"),
        filters=[loc_filter, year_filter, ('scenario', '==', rcp_scenario)],
    )
    ldipc_df = ldipc_df.drop(columns=['scenario'], errors='ignore')
    for col in [c for c in ldipc_df.columns if c not in ('location_id', 'year_id')]:
        arrays[col] = scalar_to_array(ldipc_df, col, location_ids, years)

    # Urban thresholds
    urban_files = {
        'urban_threshold_300':  Path(urban_read_path) / "urban_threshold_300.0_simple_mean.parquet",
        'urban_threshold_1500': Path(urban_read_path) / "urban_threshold_1500.0_simple_mean.parquet",
    }
    for key, path in urban_files.items():
        udf = read_parquet_with_integer_ids(str(path), filters=[loc_filter, year_filter])
        for col in [c for c in udf.columns if c not in ('location_id', 'year_id')]:
            arrays[col] = scalar_to_array(udf, col, location_ids, years)

    # Flooding
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
