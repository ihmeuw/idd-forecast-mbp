"""
Age-sex disaggregation functions for the idd-forecast-mbp pipeline.

Two structurally distinct methods are preserved — they cannot be consolidated:

  disaggregate_age_sex_malaria:
    Normalized RR-fraction method. Distributes all-age counts using
    population-weighted relative-risk fractions. Canonical source:
    04_forecasting/as_malaria_fractions.py

  disaggregate_age_sex_dengue:
    Log-rate + CFR method. Computes age-sex-specific counts directly from
    a log-rate prediction and a CFR without an all-age input.
    Canonical source: 04_forecasting/as_dengue_shifts.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Malaria — normalized RR-fraction method
# ---------------------------------------------------------------------------

def disaggregate_age_sex_malaria(
    forecast_df: pd.DataFrame,
    inc_count_col: str = 'aa_malaria_inc_count',
    mort_count_col: str = 'aa_malaria_mort_count',
    rr_inc_col: str = 'rr_inc_as',
    rr_mort_col: str = 'rr_mort_as',
    population_col: str = 'population',
    zero_age_group_id: int = 2,
) -> pd.DataFrame:
    """Disaggregate all-age malaria counts to age-sex strata using RR fractions.

    For each (location_id, year_id) group:
      1. Compute population-weighted RR: rr_pop = rr * population
      2. Normalize: fraction = rr_pop / sum(rr_pop)
      3. Multiply: as_count = fraction * aa_count
    Then zero out age_group_id == zero_age_group_id (neonates not at risk).

    Parameters
    ----------
    forecast_df:
        DataFrame with all-age count columns, age/sex-specific RR columns,
        and population. Must have 'location_id', 'year_id', 'age_group_id'.
    inc_count_col:
        All-age incidence count column name.
    mort_count_col:
        All-age mortality count column name.
    rr_inc_col:
        Age-sex relative risk column for incidence.
    rr_mort_col:
        Age-sex relative risk column for mortality.
    population_col:
        Population column name.
    zero_age_group_id:
        Age group ID to zero out after disaggregation. Default 2 (early neonatal).

    Returns
    -------
    forecast_df with 'malaria_inc_count_pred' and 'malaria_mort_count_pred' added.
    Intermediate columns (rr_*_pop, sum_rr_*_pop, *_fraction) are dropped.

    # Extracted from: 04_forecasting/as_malaria_fractions.py:236-255
    """
    df = forecast_df.copy()

    df['rr_inc_as_pop'] = df[rr_inc_col] * df[population_col]
    df['rr_mort_as_pop'] = df[rr_mort_col] * df[population_col]

    df['sum_rr_inc_as_pop'] = (
        df.groupby(['location_id', 'year_id'])['rr_inc_as_pop'].transform('sum')
    )
    df['sum_rr_mort_as_pop'] = (
        df.groupby(['location_id', 'year_id'])['rr_mort_as_pop'].transform('sum')
    )

    df['inc_fraction'] = df['rr_inc_as_pop'] / df['sum_rr_inc_as_pop']
    df['mort_fraction'] = df['rr_mort_as_pop'] / df['sum_rr_mort_as_pop']

    df['malaria_inc_count_pred'] = df['inc_fraction'] * df[inc_count_col]
    df['malaria_mort_count_pred'] = df['mort_fraction'] * df[mort_count_col]

    df.loc[df['age_group_id'] == zero_age_group_id, 'malaria_inc_count_pred'] = 0
    df.loc[df['age_group_id'] == zero_age_group_id, 'malaria_mort_count_pred'] = 0

    df = df.drop(columns=[
        'rr_inc_as_pop', 'rr_mort_as_pop',
        'sum_rr_inc_as_pop', 'sum_rr_mort_as_pop',
        'inc_fraction', 'mort_fraction',
    ])

    return df


# ---------------------------------------------------------------------------
# Dengue — log-rate + CFR method
# ---------------------------------------------------------------------------

def disaggregate_age_sex_dengue(
    forecast_df: pd.DataFrame,
    log_inc_rate_col: str = 'base_log_dengue_inc_rate_pred',
    cfr_col: str = 'dengue_cfr_pred',
    rr_inc_col: str = 'rr_inc_as',
    population_col: str = 'population',
) -> pd.DataFrame:
    """Compute age-sex dengue counts from a log-rate prediction and CFR.

    Counts are computed directly — no all-age input required:
      inc_count = population * exp(log_inc_rate) * rr_inc_as
      mort_count = inc_count * cfr

    Parameters
    ----------
    forecast_df:
        DataFrame with log-rate, CFR, age-sex RR, and population columns.
    log_inc_rate_col:
        Log incidence rate column (base prediction, before age-sex shift).
    cfr_col:
        Case fatality rate column (already raked / post-logit-shift).
    rr_inc_col:
        Age-sex relative risk for incidence.
    population_col:
        Population column name.

    Returns
    -------
    forecast_df with 'dengue_inc_count_pred' and 'dengue_mort_count_pred' added.

    # Extracted from: 04_forecasting/as_dengue_shifts.py:202-203
    """
    df = forecast_df.copy()

    df['dengue_inc_count_pred'] = (
        df[population_col] * np.exp(df[log_inc_rate_col]) * df[rr_inc_col]
    )
    df['dengue_mort_count_pred'] = df['dengue_inc_count_pred'] * df[cfr_col]

    return df
