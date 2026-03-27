"""
Raking and aggregation functions — delegates to lib/processing/.

Functions that were exact extractions are replaced with imports from lib.
check_concordance and make_aa_rate_variable are retained here (not in lib).
"""

import pandas as pd
import numpy as np

from idd_forecast_mbp.lib.processing._helpers import make_aa_df_square, prep_df
from idd_forecast_mbp.lib.processing.raking import (
    rake_level,
    rake_aa_count_lsae_to_gbd,
)
from idd_forecast_mbp.lib.processing.aggregation import (
    aggregate_level,
    aggregate_aa_count_lsae_to_gbd,
    aggregate_aa_rate_lsae_to_gbd,
)
from idd_forecast_mbp.lib.io.parquet import write_parquet


def check_concordance(variable, aa_full_df, aa_gbd_df, tolerance=0.01):
    aa_gbd_df = aa_gbd_df.rename(columns={variable: f'gbd_{variable}'})
    combined_df = pd.merge(aa_full_df, aa_gbd_df, on=['location_id', 'year_id'], how='inner')
    combined_df['concordance'] = (combined_df[variable] - combined_df[f'gbd_{variable}']).abs()
    combined_df = combined_df.sort_values(by='concordance', ascending=False).reset_index(drop=True)
    concordance_stats = {}
    if combined_df['concordance'].max() > tolerance:
        print(f'Warning: Maximum absolute difference for {variable} exceeds tolerance of {tolerance}: {combined_df["concordance"].max()}')
        concordance_stats = {
            'mean_absolute_diff': combined_df['concordance'].mean(),
            'median_absolute_diff': combined_df['concordance'].median(),
            'max_absolute_diff': combined_df['concordance'].max(),
            'std_absolute_diff': combined_df['concordance'].std(),
            'mean_relative_diff_pct': (combined_df['concordance'] / combined_df[f'gbd_{variable}']).mean() * 100,
            'pearson_correlation': combined_df[variable].corr(combined_df[f'gbd_{variable}']),
            'within_1_pct': (combined_df['concordance'] / combined_df[f'gbd_{variable}'] <= 0.01).mean(),
            'within_5_pct': (combined_df['concordance'] / combined_df[f'gbd_{variable}'] <= 0.05).mean(),
            'within_10_pct': (combined_df['concordance'] / combined_df[f'gbd_{variable}'] <= 0.10).mean(),
            'p95_absolute_diff': combined_df['concordance'].quantile(0.95),
            'p99_absolute_diff': combined_df['concordance'].quantile(0.99),
        }
        print(combined_df.head(10))
    else:
        print(f'✅ {variable} concordance check passed: max absolute difference {combined_df["concordance"].max()} within tolerance {tolerance}')
        concordance_stats = {}
    return concordance_stats


def make_aa_rate_variable(count_variable, aa_full_count_df, aa_full_population_df, aa_full_rate_df_path, return_full_df=False):
    rate_variable = count_variable.replace('count', 'rate')
    aa_full_rate_df = aa_full_count_df.merge(aa_full_population_df, on=['location_id', 'year_id'], how='left')
    aa_full_rate_df[rate_variable] = aa_full_rate_df[count_variable] / aa_full_rate_df['population']
    aa_full_rate_df = aa_full_rate_df.drop(columns=['population', count_variable])
    if aa_full_rate_df_path is not None:
        write_parquet(aa_full_rate_df, aa_full_rate_df_path)
    if return_full_df:
        return aa_full_rate_df


def make_aa_full_rate_df_from_aa_count_df(rate_variable, count_variable, aa_full_count_df, aa_full_population_df, aa_full_rate_df_path=None, return_full_df=False):
    if 'population' in aa_full_count_df.columns:
        aa_full_count_df = aa_full_count_df.drop(columns=['population'])
    aa_full_rate_df = aa_full_count_df.merge(
        aa_full_population_df[['location_id', 'year_id', 'population']],
        on=['location_id', 'year_id'],
        how='left',
    ).copy()
    aa_full_rate_df[rate_variable] = aa_full_rate_df[count_variable] / aa_full_rate_df['population']
    aa_full_rate_df.loc[aa_full_rate_df['population'] == 0, rate_variable] = 0
    aa_full_rate_df = aa_full_rate_df.drop(columns=[count_variable])
    if 'level' in aa_full_rate_df.columns:
        aa_full_rate_df = aa_full_rate_df.drop(columns=['level'])
    if aa_full_rate_df_path is not None:
        write_parquet(aa_full_rate_df, aa_full_rate_df_path)
    if return_full_df:
        return aa_full_rate_df
