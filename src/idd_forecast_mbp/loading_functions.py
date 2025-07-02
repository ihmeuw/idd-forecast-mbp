import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import os
import time
import xarray as xr
import tempfile

from idd_forecast_mbp import constants as rfc

#############################
# YAML
#############################
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


def read_netcdf_with_integer_ids(path, **kwargs):
    '''Read a NetCDF file and ensure ID coordinates are integers.'''
    ds = xr.open_dataset(path, **kwargs)
    ds = sort_id_coordinates(ds)
    return ensure_id_coordinates_are_integers(ds)


def write_netcdf(ds, filepath, max_retries=3, engine='netcdf4', **kwargs):
    '''
    Write NetCDF file with validation and retry logic.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset to write
    filepath : str
        Path to write the file
    max_retries : int
        Number of retry attempts
    engine : str
        NetCDF engine to use ('netcdf4', 'h5netcdf', 'scipy')
    **kwargs : dict
        Additional arguments passed to to_netcdf()
    '''
    
    for attempt in range(max_retries):
        try:
            # Write to temporary file first
            temp_dir = os.path.dirname(filepath)
            with tempfile.NamedTemporaryFile(suffix='.nc', dir=temp_dir, delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Write the data
            ds.to_netcdf(temp_path, engine=engine, **kwargs)
            
            # Validate the written file
            try:
                # Test read the entire file
                test_ds = xr.open_dataset(temp_path)
                
                # Basic validation checks
                if test_ds.sizes != ds.sizes:
                    raise ValueError(f'Dimension size mismatch: {test_ds.sizes} vs {ds.sizes}')
                
                if list(test_ds.data_vars) != list(ds.data_vars):
                    raise ValueError(f'Data variable mismatch: {list(test_ds.data_vars)} vs {list(ds.data_vars)}')
                
                if list(test_ds.coords) != list(ds.coords):
                    raise ValueError(f'Coordinate mismatch: {list(test_ds.coords)} vs {list(ds.coords)}')
                
                # Check data variable shapes
                for var in ds.data_vars:
                    if test_ds[var].shape != ds[var].shape:
                        raise ValueError(f'Shape mismatch for {var}: {test_ds[var].shape} vs {ds[var].shape}')
                
                print(f'✅ Validation passed for {filepath}')
                
                # Close the test dataset before moving
                test_ds.close()
                
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


def sort_id_coordinates(ds):
    '''Sort dataset by ID coordinates in a consistent order.'''
    id_coords = [coord for coord in ds.coords if coord.endswith('_id')]
    
    # Sort by each ID coordinate
    for coord in sorted(id_coords):
        if coord in ds.coords:
            ds = ds.sortby(coord)
    
    return ds


def ensure_id_coordinates_are_integers(ds):
    '''Ensure ID coordinates are integer types.'''
    ds_copy = ds.copy()
    
    id_coords = [coord for coord in ds_copy.coords if coord.endswith('_id')]
    
    for coord in id_coords:
        if coord in ds_copy.coords:
            # Get current coordinate values
            coord_values = ds_copy.coords[coord].values
            
            # Convert to appropriate integer type based on range
            max_val = coord_values.max()
            min_val = coord_values.min()
            
            if min_val >= 0:
                if max_val <= 255:
                    dtype = 'uint8'
                elif max_val <= 65535:
                    dtype = 'uint16'
                elif max_val <= 4294967295:
                    dtype = 'uint32'
                else:
                    dtype = 'uint64'
            else:
                if min_val >= -128 and max_val <= 127:
                    dtype = 'int8'
                elif min_val >= -32768 and max_val <= 32767:
                    dtype = 'int16'
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    dtype = 'int32'
                else:
                    dtype = 'int64'
            
            # Update coordinate with integer type
            ds_copy.coords[coord] = ds_copy.coords[coord].astype(dtype)
    
    return ds_copy


# Additional utility functions for NetCDF/xarray workflows

def optimize_netcdf_encoding(ds, compression_level=4):
    '''
    Add optimal encoding for NetCDF files to reduce file size.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset to optimize
    compression_level : int
        Compression level (1-9, higher = more compression)
    '''
    encoding = {}
    
    # Set encoding for coordinates
    for coord in ds.coords:
        if coord.endswith('_id'):
            encoding[coord] = {
                'zlib': True, 
                'complevel': compression_level,
                'shuffle': True
            }
    
    # Set encoding for data variables
    for var in ds.data_vars:
        var_encoding = {
            'zlib': True,
            'complevel': compression_level,
            'shuffle': True
        }
        
        # Add chunking for large arrays
        if ds[var].size > 1000000:  # 1M elements
            chunks = []
            for dim in ds[var].dims:
                dim_size = ds.sizes[dim]
                chunk_size = min(1000, dim_size)  # Max 1000 per dimension
                chunks.append(chunk_size)
            var_encoding['chunksizes'] = tuple(chunks)
        
        encoding[var] = var_encoding
    
    return encoding


def write_netcdf_optimized(ds, filepath, max_retries=3, compression_level=4, **kwargs):
    '''
    Write NetCDF file with optimized encoding and validation.
    '''
    # Get optimized encoding
    encoding = optimize_netcdf_encoding(ds, compression_level)
    
    # Merge with any user-provided encoding
    if 'encoding' in kwargs:
        encoding.update(kwargs['encoding'])
    kwargs['encoding'] = encoding
    
    return write_netcdf(ds, filepath, max_retries=max_retries, **kwargs)