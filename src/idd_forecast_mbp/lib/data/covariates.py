"""
Covariate loading utilities for the idd-forecast-mbp pipeline.

Covers all model-predictor covariate loading: climate (draw-specific), income, and urban.
All three follow the same dict-of-paths → read → merge pattern.

Extracted from:
  load_covariates_for_draw: 02_data_prep/forecasted_draw_specific_malaria_dataframes.py:271
                            02_data_prep/forecasted_draw_specific_dengue_dataframes.py:118
  read_income_paths:        helper_functions.py:67
  read_urban_paths:         helper_functions.py:77
  merge_dataframes:         helper_functions.py:62

Behavior changes:
  - load_covariates_for_draw: RH clip is now applied universally via
    UNIVERSAL_COVARIATE_CLIP_RULES. Malaria scripts were missing this clip;
    dengue applied it inline. Decision recorded 2026-03-27.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids


# ---------------------------------------------------------------------------
# Universal covariate clip rules
# ---------------------------------------------------------------------------

# Applied to all diseases by default in load_covariates_for_draw.
# Add future universal rules here; use extra_clip_rules for disease-specific additions.
UNIVERSAL_COVARIATE_CLIP_RULES: dict[str, tuple[float, float]] = {
    'relative_humidity': (0.001, 99.999),
}


# ---------------------------------------------------------------------------
# Climate / draw-specific covariates
# ---------------------------------------------------------------------------

def load_covariates_for_draw(
    covariate_paths: dict[str, str],
    draw: str,
    ssp_scenario: str,
    climate_data_path: str | Path,
    filters: list | None = None,
    clip_rules: dict[str, tuple[float, float]] = UNIVERSAL_COVARIATE_CLIP_RULES,
    extra_clip_rules: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Load one draw-column per covariate file and return a merged covariate DataFrame.

    For each key/path_template in covariate_paths:
      1. Formats the path with CLIMATE_DATA_PATH and ssp_scenario.
      2. Reads only location_id, year_id, and the draw column.
      3. Applies clip bounds from clip_rules and extra_clip_rules (if the key matches).
      4. Renames the draw column to the covariate key.
    Returns a single DataFrame with location_id, year_id, and all covariate columns,
    ready to merge onto a forecast DataFrame.

    Parameters
    ----------
    covariate_paths:
        Dict mapping covariate name → path template string.
        Template must accept {CLIMATE_DATA_PATH} and {ssp_scenario} via .format().
    draw:
        Draw identifier string (e.g. 'draw_0'). Used as the column name to read.
    ssp_scenario:
        SSP scenario string (e.g. 'ssp245'). Substituted into path templates.
    climate_data_path:
        Root climate data directory. Substituted as {CLIMATE_DATA_PATH} in templates.
    filters:
        Optional list of parquet filters applied when reading each covariate file.
    clip_rules:
        Dict mapping covariate key → (lower, upper) clip bounds. Applied to the
        covariate column after renaming. Defaults to UNIVERSAL_COVARIATE_CLIP_RULES,
        which clips relative_humidity to (0.001, 99.999) for all diseases.
        Pass {} to disable all clipping.
    extra_clip_rules:
        Disease-specific clip rules merged on top of clip_rules.
        Final rules = {**clip_rules, **(extra_clip_rules or {})}.

    # Extracted from: 02_data_prep/forecasted_draw_specific_malaria_dataframes.py:271
    #                 02_data_prep/forecasted_draw_specific_dengue_dataframes.py:118
    # Behavior change: clip_rules applied universally (malaria was missing RH clip).
    """
    climate_data_path = str(climate_data_path)
    applied_rules = {**clip_rules, **(extra_clip_rules or {})}

    result_df: pd.DataFrame | None = None

    for key, path_template in covariate_paths.items():
        path = path_template.format(
            CLIMATE_DATA_PATH=climate_data_path,
            ssp_scenario=ssp_scenario,
        )
        columns_to_read = ['location_id', 'year_id', draw]
        df = read_parquet_with_integer_ids(path, columns=columns_to_read, filters=filters)
        df = df.rename(columns={draw: key})

        if key in applied_rules:
            lo, hi = applied_rules[key]
            df[key] = df[key].clip(lower=lo, upper=hi)

        if result_df is None:
            result_df = df
        else:
            result_df = pd.merge(result_df, df, on=['location_id', 'year_id'], how='left')

    if result_df is None:
        return pd.DataFrame(columns=['location_id', 'year_id'])

    return result_df


# ---------------------------------------------------------------------------
# Income covariates
# ---------------------------------------------------------------------------

def read_income_paths(
    income_paths: dict[str, str],
    rcp_scenario: str,
    variable_data_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load income covariate files, filter to rcp_scenario, drop scenario column.

    Parameters
    ----------
    income_paths:
        Dict mapping covariate name → path template string.
        Template must accept {VARIABLE_DATA_PATH} via .format().
    rcp_scenario:
        RCP scenario string to filter on (keeps rows where 'scenario' == rcp_scenario).
    variable_data_path:
        Root variable data directory. Substituted as {VARIABLE_DATA_PATH} in templates.

    Returns
    -------
    Dict mapping covariate name → DataFrame (without the 'scenario' column).

    # Extracted from: helper_functions.py:67
    """
    variable_data_path = str(variable_data_path)
    result = {}
    for key, path in income_paths.items():
        path = path.format(VARIABLE_DATA_PATH=variable_data_path)
        df = read_parquet_with_integer_ids(path)
        df = df[df['scenario'] == rcp_scenario]
        df = df.drop(columns=['scenario'], errors='ignore')
        result[key] = df
    return result


# ---------------------------------------------------------------------------
# Urban covariates
# ---------------------------------------------------------------------------

def read_urban_paths(
    urban_paths: dict[str, str],
    variable_data_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load urban covariate files with column name normalization.

    Applies the following normalizations to column names:
      '300.0_simple_mean'  → '300'
      '1500.0_simple_mean' → '1500'
      '100m_urban'         → 'urban_100m'
      '1km_urban'          → 'urban_1km'
      'weighted_*'         → '*'  (strips 'weighted_' prefix)

    Also drops the 'population' column if present.

    Parameters
    ----------
    urban_paths:
        Dict mapping covariate name → path template string.
        Template must accept {VARIABLE_DATA_PATH} via .format().
    variable_data_path:
        Root variable data directory. Substituted as {VARIABLE_DATA_PATH} in templates.

    Returns
    -------
    Dict mapping covariate name → normalized DataFrame.

    # Extracted from: helper_functions.py:77
    """
    variable_data_path = str(variable_data_path)
    result = {}
    for key, path in urban_paths.items():
        path = path.format(VARIABLE_DATA_PATH=variable_data_path)
        df = pd.read_parquet(path)
        df = df.drop(columns=['population'], errors='ignore')
        df = df.rename(columns=lambda x: x.replace('300.0_simple_mean', '300') if '300.0_simple_mean' in x else x)
        df = df.rename(columns=lambda x: x.replace('1500.0_simple_mean', '1500') if '1500.0_simple_mean' in x else x)
        df = df.rename(columns=lambda x: x.replace('100m_urban', 'urban_100m') if '100m_urban' in x else x)
        df = df.rename(columns=lambda x: x.replace('1km_urban', 'urban_1km') if '1km_urban' in x else x)
        df = df.rename(columns=lambda x: x.replace('weighted_', '') if x.startswith('weighted_') else x)
        result[key] = df
    return result


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge_dataframes(
    model_df: pd.DataFrame,
    dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Left-merge a dict of DataFrames onto model_df on ['location_id', 'year_id'].

    Column name collisions are resolved by suffixing with _{key}.
    Typically used after read_income_paths() or read_urban_paths() to merge
    covariate DataFrames onto a forecast DataFrame.

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
