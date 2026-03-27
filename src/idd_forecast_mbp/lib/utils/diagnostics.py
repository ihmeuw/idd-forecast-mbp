"""
Diagnostic utilities for the idd-forecast-mbp pipeline.

check_concordance: validates that two DataFrames agree within a tolerance.
"""

from __future__ import annotations

import pandas as pd


def check_concordance(
    variable: str,
    aa_full_df: pd.DataFrame,
    aa_gbd_df: pd.DataFrame,
    tolerance: float = 0.01,
) -> dict:
    """Compare variable values between a full hierarchy df and a GBD reference df.

    Computes absolute differences and prints a warning with summary stats if any
    location/year pair exceeds tolerance. Prints a pass message otherwise.

    Parameters
    ----------
    variable:
        Name of the column to compare.
    aa_full_df:
        Full-hierarchy DataFrame containing variable.
    aa_gbd_df:
        GBD reference DataFrame containing variable.
    tolerance:
        Maximum acceptable absolute difference. Default 0.01.

    Returns
    -------
    dict with concordance statistics if tolerance is exceeded, else empty dict.

    # Extracted from: rake_and_aggregate_functions.py:6
    """
    aa_gbd_df = aa_gbd_df.rename(columns={variable: f'gbd_{variable}'})
    combined_df = pd.merge(aa_full_df, aa_gbd_df, on=['location_id', 'year_id'], how='inner')
    combined_df['concordance'] = (combined_df[variable] - combined_df[f'gbd_{variable}']).abs()
    combined_df = combined_df.sort_values(by='concordance', ascending=False).reset_index(drop=True)

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
