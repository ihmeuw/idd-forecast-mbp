import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import os
import time
import xarray as xr
import tempfile
from functools import reduce
import operator

from idd_forecast_mbp import constants as rfc

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


def write_parquet(df, filepath, max_retries=3, validate=True, validation_method='metadata', overwrite=True,
                  compression='lz4', index=False, use_atomic=False, row_group_size=100000, **kwargs):
    '''
    Write parquet file with memory-efficient validation options for large dataframes.
    
    Parameters:
    - validate: Whether to validate the written file
    - validation_method: 
        - 'none': No validation (fastest)
        - 'metadata': Validate row count and columns without loading data (low memory)
        - 'sample': Validate a small sample of rows (moderate memory)
        - 'full': Validate by reading entire file back (high memory, not recommended for large files)
    - use_atomic: If True, uses temporary file approach (safer but uses 2x disk space)
    - row_group_size: Number of rows per parquet row group (smaller = better for partial reads)
    '''
    import os
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    
    # Delete existing file first if overwrite is True
    if overwrite and os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f'🗑️ Removed existing file: {filepath}')
        except Exception as e:
            print(f'⚠️ Warning: Could not remove existing file {filepath}: {e}')

    # For very large dataframes, override validation to metadata-only if set to full
    df_size_gb = df.memory_usage(deep=True).sum() / (1024**3)
    if df_size_gb > 10 and validation_method == 'full':
        print(f"⚠️ DataFrame is {df_size_gb:.1f}GB. Switching to metadata-only validation.")
        validation_method = 'metadata'
    
    for attempt in range(max_retries):
        try:
            target_path = filepath
            if use_atomic:
                temp_dir = os.path.dirname(filepath)
                with tempfile.NamedTemporaryFile(suffix='.parquet', dir=temp_dir, delete=False) as tmp_file:
                    target_path = tmp_file.name
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Write to target path
            df.to_parquet(
                target_path, 
                compression=compression, 
                index=index,
                row_group_size=row_group_size,
                **kwargs
            )
            # Set file permissions to let anyone do anything with the file
            os.chmod(target_path, 0o775)
            
            # Validate based on selected method
            if validate:
                if validation_method == 'none':
                    pass  # No validation
                
                elif validation_method == 'metadata':
                    # Low memory validation - only checks metadata
                    metadata = pq.read_metadata(target_path)
                    if metadata.num_rows != len(df):
                        raise ValueError(f'Row count mismatch: {metadata.num_rows} vs {len(df)}')
                    
                    # Optionally check schema (column names)
                    parquet_schema = pq.read_schema(target_path)
                    parquet_columns = set(parquet_schema.names)
                    df_columns = set(df.columns)
                    if parquet_columns != df_columns:
                        raise ValueError(f'Column mismatch: {parquet_columns} vs {df_columns}')
                    
                    print(f'✅ Metadata validation passed for {filepath}')
                
                elif validation_method == 'sample':
                    # Sample validation - reads first and last N rows
                    sample_size = min(1000, len(df) // 100)  # 1% of rows up to 1000
                    
                    # Read first N rows
                    first_rows = pd.read_parquet(target_path, nrows=sample_size)
                    if len(first_rows) != min(sample_size, len(df)):
                        raise ValueError(f'Sample row count mismatch')
                    
                    # Check columns match
                    if set(first_rows.columns) != set(df.columns):
                        raise ValueError(f'Column mismatch in sample')
                    
                    print(f'✅ Sample validation passed for {filepath}')
                
                elif validation_method == 'full':
                    # Full validation - reads entire file back (memory intensive)
                    test_df = pd.read_parquet(target_path)
                    if len(test_df) != len(df):
                        raise ValueError(f'Row count mismatch: {len(test_df)} vs {len(df)}')
                    if list(test_df.columns) != list(df.columns):
                        raise ValueError('Column mismatch')
                    print(f'✅ Full validation passed for {filepath}')
            
            # Move temp file to final location if using atomic write
            if use_atomic:
                os.rename(target_path, filepath)
            
            return True
                    
        except Exception as e:
            print(f'Attempt {attempt + 1} failed: {e}')
            
            # Clean up on failure
            if use_atomic and 'target_path' in locals() and os.path.exists(target_path):
                os.remove(target_path)
            elif not use_atomic and os.path.exists(filepath):
                os.remove(filepath)  # Remove partial file
            
            if attempt == max_retries - 1:
                print(f'❌ Failed to write {filepath} after {max_retries} attempts')
                raise e
            else:
                print(f'Retrying... ({attempt + 1}/{max_retries})')
                
    return False


def filter_df(df, **id_filters):
    # Validate columns
    for id_column in id_filters.keys():
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found in dataframe")
    
    if not id_filters:
        return df
    
    # Build boolean masks
    conditions = []
    for id_column, id_values in id_filters.items():
        if isinstance(id_values, (int, float, str)):
            conditions.append(df[id_column] == id_values)
        else:
            conditions.append(df[id_column].isin(id_values))
    
    # Combine all conditions with AND using reduce
    combined_mask = reduce(operator.and_, conditions)
    return df[combined_mask]

def filter_df_by_range(df, **column_ranges):
    """
    Filter DataFrame by column ranges.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to filter
    **column_ranges : keyword arguments
        Each keyword should be a column name with a tuple (min, max) for the range
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    
    Examples:
    ---------
    # Filter by year range and age range
    filtered_df = filter_df_by_range(
        df, 
        year_id=(2020, 2022),
        age_group_id=(5, 15),
        aa_malaria_mort_count=(50, 200)
    )
    
    # Single range filter
    filtered_df = filter_df_by_range(df, population=(8000, 12000))
    """
    from functools import reduce
    import operator
    
    # Validate columns
    for column_name in column_ranges.keys():
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
    
    if not column_ranges:
        return df
    
    # Build range conditions
    conditions = []
    for column_name, (min_val, max_val) in column_ranges.items():
        range_condition = (df[column_name] >= min_val) & (df[column_name] <= max_val)
        conditions.append(range_condition)
    
    # Combine all conditions with AND using reduce
    combined_mask = reduce(operator.and_, conditions)
    
    return df[combined_mask]