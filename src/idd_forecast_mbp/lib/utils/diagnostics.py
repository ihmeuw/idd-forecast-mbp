"""
Diagnostic utilities for the idd-forecast-mbp pipeline.

check_concordance: validates that two DataFrames agree within a tolerance.
check_column_for_problematic_values: checks a column for NaN, inf, negative, non-numeric values.
"""

from __future__ import annotations

import numpy as np
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


def check_column_for_problematic_values(
    column_name: str,
    df: pd.DataFrame,
    return_report: bool = False,
    verbose: bool = False,
) -> dict | None:
    """Check a DataFrame column for NaN, infinite, negative, and non-numeric values.

    Prints a summary if any problems are found or if verbose=True.

    Parameters
    ----------
    column_name:
        Name of the column to check.
    df:
        DataFrame containing the column.
    return_report:
        If True, return a dict with counts and problematic rows.
    verbose:
        If True, print detailed output even when no problems are found.

    Returns
    -------
    Report dict if return_report=True, else None.

    # Extracted from: helper_functions.py:109
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe")

    column = df[column_name]
    nan_count = column.isna().sum()
    inf_count = np.isinf(column).sum() if np.issubdtype(column.dtype, np.number) else 0
    negative_count = (column < 0).sum() if np.issubdtype(column.dtype, np.number) else 0
    non_numeric_count = 0
    if column.dtype == object:
        non_numeric_count = pd.to_numeric(column, errors='coerce').isna().sum()

    problematic_mask = column.isna()
    if np.issubdtype(column.dtype, np.number):
        problematic_mask = problematic_mask | np.isinf(column) | (column < 0)
    if column.dtype == object:
        problematic_mask = problematic_mask | pd.to_numeric(column, errors='coerce').isna()

    problematic_rows = df[problematic_mask]

    summary_df = pd.DataFrame({
        'Check Type': ['NaN Values', 'Infinite Values', 'Negative Values', 'Non-numeric Values'],
        'Count': [nan_count, inf_count, negative_count, non_numeric_count],
        'Status': [
            '✅ Pass' if nan_count == 0 else '❌ Fail',
            '✅ Pass' if inf_count == 0 else '❌ Fail',
            '✅ Pass' if negative_count == 0 else '❌ Fail',
            '✅ Pass' if non_numeric_count == 0 else '❌ Fail',
        ],
    })

    if verbose or len(problematic_rows) > 0:
        print(f"\n=== Summary for {column_name} ===")
        print(f"Data type: {column.dtype}  |  Total rows: {len(df)}  |  Problematic: {len(problematic_rows)}")
        print(summary_df.to_string(index=False))
        if len(problematic_rows) > 0:
            print("\nFirst 10 problematic rows:")
            print(problematic_rows.head(10))
            print(f"\nUnique problematic values: {problematic_rows[column_name].unique()}")

    if return_report:
        return {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'negative_count': negative_count,
            'non_numeric_count': non_numeric_count,
            'total_problematic': len(problematic_rows),
            'problematic_rows': problematic_rows,
            'data_type': str(column.dtype),
            'summary_df': summary_df,
        }

    print('✅ Pass')
    return None
