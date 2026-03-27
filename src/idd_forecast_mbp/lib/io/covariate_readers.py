"""
Covariate data reading helpers for the idd-forecast-mbp pipeline.

Extracted from: helper_functions.py
"""

from __future__ import annotations

import pandas as pd

from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids


def merge_dataframes(model_df: pd.DataFrame, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-merge a dict of DataFrames onto model_df on ['location_id', 'year_id'].

    Parameters
    ----------
    model_df:
        Base DataFrame to merge into.
    dfs:
        Dict of {key: df} where each df has 'location_id' and 'year_id'.
        Suffix '_{key}' is added to any colliding columns.

    # Extracted from: helper_functions.py:62
    """
    for key, df in dfs.items():
        model_df = pd.merge(
            model_df, df,
            on=['location_id', 'year_id'],
            how='left',
            suffixes=('', f'_{key}'),
        )
    return model_df


def read_income_paths(
    income_paths: dict[str, str],
    rcp_scenario: str,
    variable_data_path: str,
) -> dict[str, pd.DataFrame]:
    """Read income covariate parquets and filter to the given RCP scenario.

    Parameters
    ----------
    income_paths:
        Dict of {key: path_template} where path_template may contain
        '{VARIABLE_DATA_PATH}' placeholder.
    rcp_scenario:
        RCP scenario string to filter on (e.g. 'rcp45').
    variable_data_path:
        Value to substitute for '{VARIABLE_DATA_PATH}' in path templates.

    Returns
    -------
    Dict of {key: DataFrame} with 'scenario' column dropped.

    # Extracted from: helper_functions.py:67
    """
    income_dfs = {}
    for key, path in income_paths.items():
        path = path.format(VARIABLE_DATA_PATH=variable_data_path)
        df = read_parquet_with_integer_ids(path)
        df = df[df['scenario'] == rcp_scenario]
        df = df.drop(columns=['scenario'], errors='ignore')
        income_dfs[key] = df
    return income_dfs


def read_urban_paths(
    urban_paths: dict[str, str],
    variable_data_path: str,
) -> dict[str, pd.DataFrame]:
    """Read urban covariate parquets with standardized column renaming.

    Strips 'weighted_' prefix and normalizes threshold/resolution suffixes
    ('300.0_simple_mean' → '300', '1km_urban' → 'urban_1km', etc.).

    Parameters
    ----------
    urban_paths:
        Dict of {key: path_template} where path_template may contain
        '{VARIABLE_DATA_PATH}' placeholder.
    variable_data_path:
        Value to substitute for '{VARIABLE_DATA_PATH}' in path templates.

    Returns
    -------
    Dict of {key: DataFrame} with 'population' dropped and columns renamed.

    # Extracted from: helper_functions.py:77
    """
    urban_dfs = {}
    for key, path in urban_paths.items():
        path = path.format(VARIABLE_DATA_PATH=variable_data_path)
        df = pd.read_parquet(path)
        df = df.drop(columns=['population'], errors='ignore')
        df = df.rename(columns=lambda x: x.replace('300.0_simple_mean', '300') if '300.0_simple_mean' in x else x)
        df = df.rename(columns=lambda x: x.replace('1500.0_simple_mean', '1500') if '1500.0_simple_mean' in x else x)
        df = df.rename(columns=lambda x: x.replace('100m_urban', 'urban_100m') if '100m_urban' in x else x)
        df = df.rename(columns=lambda x: x.replace('1km_urban', 'urban_1km') if '1km_urban' in x else x)
        df = df.rename(columns=lambda x: x.replace('weighted_', '') if 'weighted_' in x else x)
        urban_dfs[key] = df
    return urban_dfs
