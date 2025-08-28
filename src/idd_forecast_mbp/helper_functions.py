from pathlib import Path
import pandas as pd
import numpy as np
import os
import time
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet, ensure_id_columns_are_integers, sort_id_columns

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / '02-processed_data'
hierarchy = 'lsae_1209'
VARIABLE_DATA_PATH = f'{PROCESSED_DATA_PATH}/{hierarchy}'

cause_map = rfc.cause_map
measure_map = rfc.measure_map
metric_map = rfc.metric_map

def check_folders_for_files(folders_and_files, delete_existing=True):
    """
    Check and optionally delete specific files from multiple folders.
    
    Args:
        folders_and_files: Dictionary mapping folder paths to lists of filenames
        delete_existing: If True, delete existing files; if False, just check existence
    
    Returns:
        dict: Dictionary mapping folder paths to boolean (True if all files were present)
    """
    results = {}
    for folder_path, files_to_check in folders_and_files.items():
        print(f"\nProcessing folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"  Creating new folder: {folder_path}")
            mkdir(folder_path, parents=True, exist_ok=True)
            results[folder_path] = False
            continue
        
        action = "delete" if delete_existing else "check"
        print(f"  Checking files to {action}")
        
        all_files_present = True
        for filename in files_to_check:
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                print(f"    File not found: {filename}")
                all_files_present = False
            elif delete_existing:
                print(f"    Deleting: {filename}")
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            else:
                print(f"    Found: {filename}")
        
        results[folder_path] = all_files_present
    
    return results


def merge_dataframes(model_df, dfs):
    for key, df in dfs.items():
        model_df = pd.merge(model_df, df, on=['location_id', 'year_id'], how='left', suffixes=('', f'_{key}'))
    return model_df

def read_income_paths(income_paths, rcp_scenario, VARIABLE_DATA_PATH):
    income_dfs = {}
    for key, path in income_paths.items():
        path = path.format(VARIABLE_DATA_PATH=VARIABLE_DATA_PATH)
        income_dfs[key] = read_parquet_with_integer_ids(path)
        income_dfs[key] = income_dfs[key][income_dfs[key]['scenario'] == rcp_scenario]
        # Drop scenario   
        income_dfs[key] = income_dfs[key].drop(columns=['scenario'], errors='ignore')
    return income_dfs 

def read_urban_paths(urban_paths, VARIABLE_DATA_PATH):
    urban_dfs = {}
    for key, path in urban_paths.items():
        path = path.format(VARIABLE_DATA_PATH=VARIABLE_DATA_PATH)
        urban_dfs[key] = pd.read_parquet(path)
        # Drop population
        urban_dfs[key] = urban_dfs[key].drop(columns=['population'], errors='ignore')
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace('300.0_simple_mean', '300') if '300.0_simple_mean' in x else x)
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace('1500.0_simple_mean', '1500') if '1500.0_simple_mean' in x else x)
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace('100m_urban', 'urban_100m') if '100m_urban' in x else x)
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace('1km_urban', 'urban_1km') if '1km_urban' in x else x)
        # Remove every instance of 'weighted_' from the column names
        urban_dfs[key] = urban_dfs[key].rename(columns=lambda x: x.replace('weighted_', '') if 'weighted_' in x else x)
    return urban_dfs

def level_filter(hierarchy_df, start_level, end_level=None, return_ids = False):
    """
    Returns a filter for the hierarchy DataFrame to select rows between start_level and end_level.
    """
    if end_level is None:
        end_level = start_level
    levels_to_filter_on = list(range(start_level, end_level + 1))
    location_ids = (
        hierarchy_df[hierarchy_df['level'].isin(levels_to_filter_on)]
        ['location_id'].unique().tolist()
    )
    location_filter = ('location_id', 'in', location_ids)
    if return_ids:
        return location_filter, location_ids
    else:
        return location_filter

def check_column_for_problematic_values(column_name, df, return_report=False, verbose=False):
    """
    Check a dataframe column for problematic values (NaN, infinite, negative, non-numeric).
    For mortality columns, also provides additional statistics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the column to check
    column_name : str
        The name of the column to check
    verbose : bool, default=True
        If True, prints detailed information about the checks
        
    Returns:
    --------
    dict
        Dictionary containing counts of different problematic value types and the problematic rows
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe")
    
    column = df[column_name]
    nan_count = column.isna().sum()
    inf_count = np.isinf(column).sum()
    negative_count = (column < 0).sum()
    non_numeric_count = 0
    if column.dtype == 'object':
        non_numeric = pd.to_numeric(column, errors='coerce').isna()
        non_numeric_count = non_numeric.sum()
    problematic_mask = (
        column.isna() |
        np.isinf(column) |
        (column < 0)
    )
    if column.dtype == 'object':
        problematic_mask = problematic_mask | pd.to_numeric(column, errors='coerce').isna()
    
    problematic_rows = df[problematic_mask]
    
    summary_data = {
        'Check Type': ['NaN Values', 'Infinite Values', 'Negative Values', 'Non-numeric Values'],
        'Count': [nan_count, inf_count, negative_count, non_numeric_count],
        'Status': [
            '✅ Pass' if nan_count == 0 else '❌ Fail',
            '✅ Pass' if inf_count == 0 else '❌ Fail', 
            '✅ Pass' if negative_count == 0 else '❌ Fail',
            '✅ Pass' if non_numeric_count == 0 else '❌ Fail'
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # Print summary if verbose or if problems found
    if verbose or len(problematic_rows) > 0:
        print(f"\n=== Summary for {column_name} ===")
        print(f"Data type: {column.dtype}")
        print(f"Total rows: {len(df)}")
        print(f"Total problematic rows: {len(problematic_rows)}")
        print("\nDetailed Check Results:")
        print(summary_df.to_string(index=False))
             
        if len(problematic_rows) > 0:
            print("\nFirst 10 problematic rows:")
            print(problematic_rows.head(10))
            
            # Show unique problematic values
            print(f"\nUnique problematic values:")
            print(problematic_rows[column_name].unique())
    
    if return_report:
        report = {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'negative_count': negative_count,
            'non_numeric_count': non_numeric_count,
            'total_problematic': len(problematic_rows),
            'problematic_rows': problematic_rows,
            'data_type': str(column.dtype),
            'summary_df': summary_df
        }
        return report
    else:
        print('✅ Pass')

def verify_hdf_checksum(filepath, key='df'):
    """
    Verify the integrity of an HDF5 file using stored checksum
    
    Parameters:
    -----------
    filepath : str
        Path to the HDF5 file
    key : str, default='df'
        Key of the dataframe in the HDF5 file
        
    Returns:
    --------
    bool
        True if checksum matches, False otherwise
    """
    import json
    import hashlib
    
    checksum_file = filepath + '.checksum'
    
    if not os.path.exists(checksum_file):
        print(f"⚠️ No checksum file found for {filepath}")
        return False
    
    try:
        # Load stored checksum
        with open(checksum_file, 'r') as f:
            checksum_data = json.load(f)
        
        stored_checksum = checksum_data['checksum']
        
        # Read the file and calculate checksum
        df = pd.read_hdf(filepath, key=key)
        df_str = df.to_string(index=False).encode('utf-8')
        file_checksum = hashlib.sha256(df_str).hexdigest()
        
        if file_checksum == stored_checksum:
            print(f"✅ Checksum verification passed for {filepath}")
            return True
        else:
            print(f"❌ Checksum verification failed for {filepath}")
            print(f"   Expected: {stored_checksum}")
            print(f"   Got:      {file_checksum}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying checksum for {filepath}: {e}")
        return False
