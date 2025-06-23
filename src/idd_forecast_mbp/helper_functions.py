import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from idd_forecast_mbp import constants as rfc


PROCESSED_DATA_PATH = rfc.MODEL_ROOT / '02-processed_data'
hierarchy = 'lsae_1209'
VARIABLE_DATA_PATH = f'{PROCESSED_DATA_PATH}/{hierarchy}'

cause_map = rfc.cause_map
measure_map = rfc.measure_map
metric_map = rfc.metric_map



# Is it better to nest these functions, have the material repeated or read both in every time. Most of the time we only need the second one
def load_yaml_dictionary(yaml_path: str) -> dict:
    # Read YAML
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return(yaml_data['COVARIATE_DICT'])

def parse_yaml_dictionary(covariate: str) -> dict:
    YAML_PATH = rfc.REPO_ROOT / rfc.repo_name / 'src' / rfc.package_name /  'COVARIATE_DICT.yaml'
    # Extract covariate-specific config
    covariate_dict = load_yaml_dictionary(YAML_PATH)
    # Check if the covariate exists in the dictionary
    if covariate not in covariate_dict:
        raise ValueError(f"Covariate '{covariate}' not found in the dictionary.")
    # Extract the covariate entry
    covariate_entry = covariate_dict.get(covariate, [])

    covariate_resolution = covariate_entry['covariate_resolution_numerator'] / covariate_entry['covariate_resolution_denominator']

    years = list(range(covariate_entry['year_start'], covariate_entry['year_end'] + 1))

    # Build the return dict dynamically
    result = {
        'covariate_name': covariate_entry['covariate_name'],
        'covariate_resolution': covariate_resolution,
        'years': years,
        'synoptic': covariate_entry['synoptic'],
        'cc_sensitive': covariate_entry['cc_sensitive'],
        'summary_statistic': covariate_entry['summary_statistic'],
        'path': covariate_entry['path'],
    }

    return result

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

def ensure_id_columns_are_integers(df):
    '''
    Ensures that any column ending with '_id' is cast to integer type.
    '''
    for col in df.columns:
        if col.endswith('_id') and pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Capital I
    return df

def sort_id_columns(df):
    '''
    Sorts the DataFrame by id columns with 'location_id' and 'year_id' having priority if they exist.
    '''
    id_columns = [col for col in df.columns if col.endswith('_id')]

    # Reorder to put location_id first and year_id second
    ordered_columns = []
    if 'location_id' in id_columns:
        ordered_columns.append('location_id')
    if 'year_id' in id_columns:
        ordered_columns.append('year_id')

    # Add remaining id columns
    remaining_columns = [col for col in id_columns if col not in ['location_id', 'year_id']]
    id_columns = ordered_columns + remaining_columns
    
    if id_columns:
        df = df.sort_values(by=id_columns)

    return df

def read_parquet_with_integer_ids(path, **kwargs):
    '''Read a parquet file and ensure ID columns are integers.'''
    df = pd.read_parquet(path, **kwargs)
    df = sort_id_columns(df)
    return ensure_id_columns_are_integers(df)


def write_parquet(df, filepath, max_retries=3, compression='snappy', index=False, **kwargs):
    '''
    Write parquet file with validation and retry logic.
    '''
    import os
    import tempfile
    from pathlib import Path
    
    for attempt in range(max_retries):
        try:
            # Write to temporary file first
            temp_dir = os.path.dirname(filepath)
            with tempfile.NamedTemporaryFile(suffix='.parquet', dir=temp_dir, delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Write the data
            df.to_parquet(temp_path, compression=compression, index=index, **kwargs)
            
            # Validate the written file
            try:
                # Test read the entire file
                test_df = pd.read_parquet(temp_path)
                
                # Basic validation checks
                if len(test_df) != len(df):
                    raise ValueError(f'Row count mismatch: {len(test_df)} vs {len(df)}')
                
                if list(test_df.columns) != list(df.columns):
                    raise ValueError('Column mismatch')
                
                print(f'✅ Validation passed for {filepath}')
                
                # Move temp file to final location
                os.rename(temp_path, filepath)
                return True
                
            except Exception as e:
                print(f'❌ Validation failed for {filepath}: {e}')
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
                
        except Exception as e:
            print(f'Attempt {attempt + 1} failed: {e}')
            if attempt == max_retries - 1:
                print(f'❌ Failed to write {filepath} after {max_retries} attempts')
                raise e
            else:
                print(f'Retrying... ({attempt + 1}/{max_retries})')
                
    return False

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
        return {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'negative_count': negative_count,
            'non_numeric_count': non_numeric_count,
            'total_problematic': len(problematic_rows),
            'problematic_rows': problematic_rows,
            'data_type': str(column.dtype),
            'summary_df': summary_df
        }
    else:
        print('✅ Pass')

