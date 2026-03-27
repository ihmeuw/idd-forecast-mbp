"""
DAH scenario generation for the idd-forecast-mbp pipeline.

Extracted from: 02_data_prep/forecasted_draw_specific_malaria_dataframes.py:76
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_dah_scenarios(
    baseline_df: pd.DataFrame,
    ssp_scenario: str,
    year_start: int = 2000,
    reference_year: int = 2023,
    modification_start_year: int = 2026,
    dah_scenario_names: list[str] | None = None,
) -> tuple[list[pd.DataFrame], list[str]]:
    """Generate four DAH funding scenarios for malaria forecasting.

    Scenarios produced:
      Baseline   — original DAH projections, unchanged
      Constant   — DAH held at reference_year level for all future years
      Increasing — DAH multiplied by [1.2, 1.4, 1.6, 1.8, 2.0] over 5 years
                   then held at 2.0× thereafter
      Decreasing — DAH multiplied by [0.8, 0.6, 0.4, 0.2, 0.0] over 5 years
                   then held at 0× thereafter

    Parameters
    ----------
    baseline_df:
        DataFrame with columns: 'location_id', 'year_id', 'A0_location_id',
        'aa_population', 'mal_DAH_total_per_capita'. Rows with year_id <
        year_start are filtered out before processing.
    ssp_scenario:
        SSP scenario label (e.g. 'ssp245'). Written to 'ssp_scenario' column.
    year_start:
        Minimum year to retain from baseline_df. Default 2000.
    reference_year:
        Year whose DAH level is used for the Constant scenario. Default 2023.
    modification_start_year:
        First year the increasing/decreasing multipliers are applied. Default 2026.
    dah_scenario_names:
        Override default scenario names. Default ['Baseline', 'Constant',
        'Increasing', 'Decreasing'].

    Returns
    -------
    (dah_scenarios, dah_scenario_names)
        dah_scenarios: list of four DataFrames (Baseline, Constant, Increasing,
        Decreasing), each with a 'dah_scenario' column set to the scenario name.
        dah_scenario_names: list of scenario name strings.

    # Extracted from: 02_data_prep/forecasted_draw_specific_malaria_dataframes.py:76
    """
    if dah_scenario_names is None:
        dah_scenario_names = ['Baseline', 'Constant', 'Increasing', 'Decreasing']

    increasing_factors = {
        modification_start_year:     1.2,
        modification_start_year + 1: 1.4,
        modification_start_year + 2: 1.6,
        modification_start_year + 3: 1.8,
        modification_start_year + 4: 2.0,
    }
    decreasing_factors = {
        modification_start_year:     0.8,
        modification_start_year + 1: 0.6,
        modification_start_year + 2: 0.4,
        modification_start_year + 3: 0.2,
        modification_start_year + 4: 0.0,
    }

    # --- Baseline ---
    baseline_df = baseline_df.copy()
    baseline_df = baseline_df[baseline_df['year_id'] >= year_start]
    baseline_df['A0_location_id'] = baseline_df['A0_location_id'].astype(int)
    baseline_df['A0_af'] = 'A0_' + baseline_df['A0_location_id'].astype(str)
    baseline_df['ssp_scenario'] = ssp_scenario
    baseline_df['dah_scenario'] = 'Baseline'

    # --- Scenario 1: Constant ---
    scenario_1_df = baseline_df.copy()
    scenario_1_df['mal_DAH_total'] = (
        scenario_1_df['mal_DAH_total_per_capita'] * scenario_1_df['aa_population']
    )
    values_ref_year = scenario_1_df[
        scenario_1_df['year_id'] == reference_year
    ][['location_id', 'mal_DAH_total']]
    scenario_1_df = scenario_1_df.merge(
        values_ref_year, on='location_id', suffixes=('', f'_{reference_year}')
    )
    mask = scenario_1_df['year_id'] >= reference_year + 1
    scenario_1_df.loc[mask, 'mal_DAH_total'] = scenario_1_df.loc[mask, f'mal_DAH_total_{reference_year}']
    scenario_1_df = scenario_1_df.drop(columns=f'mal_DAH_total_{reference_year}')
    scenario_1_df['mal_DAH_total_per_capita'] = (
        scenario_1_df['mal_DAH_total'] / scenario_1_df['aa_population']
    )
    scenario_1_df['log_mal_DAH_total_per_capita'] = np.log(
        scenario_1_df['mal_DAH_total_per_capita'] + 1e-6
    )
    scenario_1_df['ssp_scenario'] = ssp_scenario
    scenario_1_df['dah_scenario'] = 'Constant'

    # --- Scenario 2: Increasing ---
    scenario_2_df = baseline_df.copy()
    for year, factor in increasing_factors.items():
        mask = scenario_2_df['year_id'] == year
        scenario_2_df.loc[mask, 'mal_DAH_total_per_capita'] *= factor
        scenario_2_df.loc[mask, 'mal_DAH_total'] = (
            scenario_2_df.loc[mask, 'mal_DAH_total_per_capita'] *
            scenario_2_df.loc[mask, 'aa_population']
        )
    max_factor = max(increasing_factors.values())
    max_year = max(increasing_factors.keys())
    mask = scenario_2_df['year_id'] > max_year
    scenario_2_df.loc[mask, 'mal_DAH_total_per_capita'] *= max_factor
    scenario_2_df.loc[mask, 'mal_DAH_total'] = (
        scenario_2_df.loc[mask, 'mal_DAH_total_per_capita'] *
        scenario_2_df.loc[mask, 'aa_population']
    )
    scenario_2_df['log_mal_DAH_total_per_capita'] = np.log(
        scenario_2_df['mal_DAH_total_per_capita'] + 1e-6
    )
    scenario_2_df['ssp_scenario'] = ssp_scenario
    scenario_2_df['dah_scenario'] = 'Increasing'

    # --- Scenario 3: Decreasing ---
    scenario_3_df = baseline_df.copy()
    for year, factor in decreasing_factors.items():
        mask = scenario_3_df['year_id'] == year
        scenario_3_df.loc[mask, 'mal_DAH_total_per_capita'] *= factor
        scenario_3_df.loc[mask, 'mal_DAH_total'] = (
            scenario_3_df.loc[mask, 'mal_DAH_total_per_capita'] *
            scenario_3_df.loc[mask, 'aa_population']
        )
    min_factor = min(decreasing_factors.values())
    max_year = max(decreasing_factors.keys())
    mask = scenario_3_df['year_id'] > max_year
    scenario_3_df.loc[mask, 'mal_DAH_total_per_capita'] = min_factor
    scenario_3_df.loc[mask, 'mal_DAH_total'] = (
        min_factor * scenario_3_df.loc[mask, 'aa_population']
    )
    scenario_3_df['log_mal_DAH_total_per_capita'] = np.log(
        scenario_3_df['mal_DAH_total_per_capita'] + 1e-6
    )
    scenario_3_df['ssp_scenario'] = ssp_scenario
    scenario_3_df['dah_scenario'] = 'Decreasing'

    dah_scenarios = [baseline_df, scenario_1_df, scenario_2_df, scenario_3_df]
    return dah_scenarios, dah_scenario_names
