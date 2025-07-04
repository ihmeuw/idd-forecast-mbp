import xarray as xr
import os
import tempfile
from pathlib import Path


# # Convert to xarray and merge
# hierarchy_ds = hierarchy_df.set_index('location_id').to_xarray()
# ds_merged = xr.merge([ds, ds_locations])




def read_netcdf_with_integer_ids(path, **kwargs):
    '''Read a NetCDF file and ensure ID coordinates are integers.'''
    ds = xr.open_dataset(path, **kwargs)
    ds = sort_id_coordinates(ds)
    return ensure_id_coordinates_are_integers(ds)


def write_netcdf(ds, filepath, max_retries=3, engine='netcdf4', 
                 compression=True, compression_level=4, chunking=True, 
                 chunk_threshold=1000000, max_chunk_size=1000, **kwargs):
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
    compression : bool
        Whether to apply compression (default True)
    compression_level : int
        Compression level 1-9, higher = more compression (default 4)
    chunking : bool
        Whether to apply chunking for large arrays (default True)
    chunk_threshold : int
        Minimum array size to trigger chunking (default 1M elements)
    max_chunk_size : int
        Maximum chunk size per dimension (default 1000)
    **kwargs : dict
        Additional arguments passed to to_netcdf()
    '''
    
    # Build encoding if compression or chunking is enabled
    encoding = {}
    if compression or chunking:
        # Set encoding for coordinates
        for coord in ds.coords:
            if coord.endswith('_id') and compression:
                encoding[coord] = {
                    'zlib': True, 
                    'complevel': compression_level,
                    'shuffle': True
                }
        
        # Set encoding for data variables
        for var in ds.data_vars:
            var_encoding = {}
            
            if compression:
                var_encoding.update({
                    'zlib': True,
                    'complevel': compression_level,
                    'shuffle': True
                })
            
            # Add chunking for large arrays
            if chunking and ds[var].size > chunk_threshold:
                chunks = []
                for dim in ds[var].dims:
                    dim_size = ds.sizes[dim]
                    chunk_size = min(max_chunk_size, dim_size)
                    chunks.append(chunk_size)
                var_encoding['chunksizes'] = tuple(chunks)
            
            if var_encoding:  # Only add if we have encoding settings
                encoding[var] = var_encoding
    
    # Merge with any user-provided encoding
    if 'encoding' in kwargs:
        encoding.update(kwargs['encoding'])
    if encoding:  # Only add encoding if we have settings
        kwargs['encoding'] = encoding
    
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
    '''
    Sorts the Dataset by id coordinates with 'location_id' and 'year_id' having priority if they exist.
    '''
    id_coords = [coord for coord in ds.coords if coord.endswith('_id')]

    # Reorder to put location_id first and year_id second
    ordered_coords = []
    if 'location_id' in id_coords:
        ordered_coords.append('location_id')
    if 'year_id' in id_coords:
        ordered_coords.append('year_id')

    # Add remaining id coordinates
    remaining_coords = [coord for coord in id_coords if coord not in ['location_id', 'year_id']]
    id_coords = ordered_coords + remaining_coords
    
    if id_coords:
        ds = ds.sortby(id_coords)

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

# Additional utility function for manual encoding optimization

def optimize_netcdf_encoding(ds, compression_level=4):
    '''
    Create optimal encoding dictionary for NetCDF files (for manual use).
    
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





import pandas as pd
import numpy as np


def convert_to_xarray(df, dimensions=None, dimension_dtypes=None, variable_dtypes=None, 
                     auto_optimize_dtypes=True, validate_dimensions=True):
    """
    Convert DataFrame to optimized xarray Dataset with configurable dimensions and dtypes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    dimensions : list, optional
        List of column names to use as dimensions (coordinates). 
        If None, auto-detects columns ending with '_id'
    dimension_dtypes : dict, optional
        Dictionary mapping dimension names to desired dtypes
        Example: {'location_id': 'int32', 'year_id': 'int16'}
    variable_dtypes : dict, optional  
        Dictionary mapping variable names to desired dtypes
        Example: {'population': 'float32', 'rate': 'float64'}
    auto_optimize_dtypes : bool, default True
        Whether to automatically optimize dtypes for dimensions and variables
        not specified in dimension_dtypes/variable_dtypes
    validate_dimensions : bool, default True
        Whether to validate that dimensions form a complete rectangular grid
        
    Returns:
    --------
    xarray.Dataset
        Converted dataset with optimized dtypes
        
    Examples:
    ---------
    # Auto-detect dimensions (columns ending with '_id')
    ds = convert_to_xarray(df)
    
    # Specify custom dimensions
    ds = convert_to_xarray(df, dimensions=['location_id', 'year_id', 'age_group_id'])
    
    # Custom dtypes
    ds = convert_to_xarray(df, 
                          dimension_dtypes={'location_id': 'int32', 'year_id': 'int16'},
                          variable_dtypes={'population': 'float32', 'rate': 'float64'})
    """
    
    df_work = df.copy()
    
    # Auto-detect dimensions if not provided
    if dimensions is None:
        dimensions = [col for col in df_work.columns if col.endswith('_id')]
        if not dimensions:
            raise ValueError("No dimensions specified and no columns ending with '_id' found")
    
    # Validate that all dimensions exist in dataframe
    missing_dims = [dim for dim in dimensions if dim not in df_work.columns]
    if missing_dims:
        raise ValueError(f"Dimensions not found in dataframe: {missing_dims}")
    
    # Get variable columns (everything not a dimension)
    variables = [col for col in df_work.columns if col not in dimensions]
    
    # Initialize dtype dictionaries if not provided
    if dimension_dtypes is None:
        dimension_dtypes = {}
    if variable_dtypes is None:
        variable_dtypes = {}
    
    # Auto-optimize dtypes for dimensions
    if auto_optimize_dtypes:
        for dim in dimensions:
            if dim not in dimension_dtypes:
                dimension_dtypes[dim] = _auto_optimize_dimension_dtype(df_work[dim])
    
    # Auto-optimize dtypes for variables  
    if auto_optimize_dtypes:
        for var in variables:
            if var not in variable_dtypes:
                variable_dtypes[var] = _auto_optimize_variable_dtype(df_work[var])
    
    # Apply dimension dtypes
    for dim, dtype in dimension_dtypes.items():
        if dim in df_work.columns:
            df_work[dim] = df_work[dim].astype(dtype)
    
    # Apply variable dtypes
    for var, dtype in variable_dtypes.items():
        if var in df_work.columns:
            df_work[var] = df_work[var].astype(dtype)
    
    # Validate complete rectangular grid if requested
    if validate_dimensions:
        _validate_rectangular_grid(df_work, dimensions)
    
    # Set multidimensional index
    df_indexed = df_work.set_index(dimensions)
    
    # Convert to xarray Dataset
    ds = df_indexed.to_xarray()
    
    return ds


def _auto_optimize_dimension_dtype(series):
    """Auto-select optimal integer dtype for dimension based on value range."""
    min_val = series.min()
    max_val = series.max()
    
    # Check for missing values
    if series.isna().any():
        # Use nullable integer types for dimensions with NaN
        if min_val >= 0:
            if max_val <= 255:
                return 'UInt8'
            elif max_val <= 65535:
                return 'UInt16' 
            elif max_val <= 4294967295:
                return 'UInt32'
            else:
                return 'UInt64'
        else:
            if min_val >= -128 and max_val <= 127:
                return 'Int8'
            elif min_val >= -32768 and max_val <= 32767:
                return 'Int16'
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return 'Int32'
            else:
                return 'Int64'
    else:
        # Use standard integer types for complete dimensions
        if min_val >= 0:
            if max_val <= 255:
                return 'uint8'
            elif max_val <= 65535:
                return 'uint16'
            elif max_val <= 4294967295:
                return 'uint32'
            else:
                return 'uint64'
        else:
            if min_val >= -128 and max_val <= 127:
                return 'int8'
            elif min_val >= -32768 and max_val <= 32767:
                return 'int16'
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return 'int32'
            else:
                return 'int64'


def _auto_optimize_variable_dtype(series):
    """Auto-select optimal dtype for variable based on content."""
    # Check if it's numeric
    if not pd.api.types.is_numeric_dtype(series):
        return series.dtype  # Keep original for non-numeric
    
    # For integer-like variables
    if pd.api.types.is_integer_dtype(series) or (series % 1 == 0).all():
        return _auto_optimize_dimension_dtype(series)
    
    # For float variables, check precision needs
    # If all values have <= 6 significant digits, float32 might be sufficient
    # Otherwise use float64
    if series.isna().all():
        return 'float32'  # Default for all-NaN series
    
    # Simple heuristic: if max absolute value is reasonable for float32 precision
    max_abs = abs(series).max()
    if max_abs < 1e6:  # Conservative threshold for float32
        return 'float32'
    else:
        return 'float64'


def _validate_rectangular_grid(df, dimensions):
    """Validate that dimensions form a complete rectangular grid (no missing combinations)."""
    expected_size = 1
    for dim in dimensions:
        expected_size *= df[dim].nunique()
    
    actual_size = len(df)
    
    if actual_size != expected_size:
        raise ValueError(
            f"Incomplete rectangular grid detected. "
            f"Expected {expected_size} rows but got {actual_size}. "
            f"This suggests missing combinations of dimension values. "
            f"Use validate_dimensions=False to skip this check."
        )


# Example usage and preset configurations
COMMON_DIMENSION_CONFIGS = {
    'as_variables': {
        'dimensions': ['location_id', 'year_id', 'age_group_id', 'sex_id'],
        'dimension_dtypes': {
            'location_id': 'int32',
            'year_id': 'int16', 
            'age_group_id': 'int16',
            'sex_id': 'int8'
        }
    },
    'aa_variables': {
        'dimensions': ['location_id', 'year_id'],
        'dimension_dtypes': {
            'location_id': 'int32',
            'year_id': 'int16'
        }
    }
}

def convert_with_preset(df, preset='as_variables', **kwargs):
    """Convert DataFrame using a preset configuration."""
    if preset not in COMMON_DIMENSION_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(COMMON_DIMENSION_CONFIGS.keys())}")
    
    config = COMMON_DIMENSION_CONFIGS[preset].copy()
    config.update(kwargs)  # Allow overriding preset values
    
    return convert_to_xarray(df, **config)


import xarray as xr
import numpy as np
from functools import reduce
import operator

def filter_ds_by_multiple_coords(ds, **coord_filters):
    """
    Filter xarray Dataset by multiple coordinate dimensions using keyword arguments.
    Similar to the pandas DataFrame filtering function but for xarray.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to filter
    **coord_filters : keyword arguments
        Each keyword should be a coordinate name with values to filter for
        
    Returns:
    --------
    xarray.Dataset
        Filtered dataset
    
    Examples:
    ---------
    # Filter by single coordinates
    filtered_ds = filter_xarray_by_multiple_coords(ds, location_id=364, year_id=2020)
    
    # Filter by multiple values per coordinate
    filtered_ds = filter_xarray_by_multiple_coords(
        ds, 
        location_id=[364, 365], 
        year_id=[2020, 2021, 2022],
        sex_id=1
    )
    """
    # Validate all coordinates exist first
    for coord_name in coord_filters.keys():
        if coord_name not in ds.coords:
            raise ValueError(f"Coordinate '{coord_name}' not found in dataset")
    
    if not coord_filters:
        return ds
    
    # Build boolean masks for each coordinate
    conditions = []
    for coord_name, coord_values in coord_filters.items():
        # Handle single value vs list of values
        if isinstance(coord_values, (int, float, str, np.integer, np.floating)):
            conditions.append(ds[coord_name] == coord_values)
        else:
            conditions.append(ds[coord_name].isin(coord_values))
    
    # Combine all conditions with AND using reduce
    combined_mask = reduce(operator.and_, conditions)
    
    # Apply the mask to filter the dataset
    return ds.where(combined_mask, drop=True)


def filter_ds_by_single_coord(ds, coord_name, coord_values):
    """
    Filter xarray Dataset by a single coordinate dimension.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to filter
    coord_name : str
        Name of the coordinate to filter on (e.g., 'location_id', 'year_id')
    coord_values : int, float, str, list, or array-like
        Single coordinate value or list of coordinate values to filter for
    
    Returns:
    --------
    xarray.Dataset
        Filtered dataset
    
    Examples:
    ---------
    # Filter for single location
    filtered_ds = filter_xarray_by_single_coord(ds, 'location_id', 364)
    
    # Filter for multiple years
    filtered_ds = filter_xarray_by_single_coord(ds, 'year_id', [2020, 2021, 2022])
    """
    if coord_name not in ds.coords:
        raise ValueError(f"Coordinate '{coord_name}' not found in dataset")
    
    # Handle single value vs list of values
    if isinstance(coord_values, (int, float, str, np.integer, np.floating)):
        return ds.sel({coord_name: coord_values})
    else:
        return ds.sel({coord_name: coord_values})


def filter_ds_with_sel(ds, **coord_filters):
    """
    Alternative filtering method using xarray's .sel() method.
    This is often more efficient for exact coordinate matches.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to filter
    **coord_filters : keyword arguments
        Each keyword should be a coordinate name with values to filter for
        
    Returns:
    --------
    xarray.Dataset
        Filtered dataset
    
    Examples:
    ---------
    # This works best when you want exact coordinate matches
    filtered_ds = filter_xarray_with_sel(
        ds, 
        location_id=[364, 365], 
        year_id=[2020, 2021, 2022]
    )
    """
    # Validate coordinates
    for coord_name in coord_filters.keys():
        if coord_name not in ds.coords:
            raise ValueError(f"Coordinate '{coord_name}' not found in dataset")
    
    if not coord_filters:
        return ds
    
    # Use sel for exact coordinate selection
    return ds.sel(coord_filters)


def get_unique_coords(ds, coord_names=None):
    """
    Get unique values for coordinate dimensions to help with filtering.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to analyze
    coord_names : list, optional
        List of coordinate names. If None, will find coordinates ending with '_id'
        
    Returns:
    --------
    dict
        Dictionary with coordinate names as keys and unique values as values
        
    Examples:
    ---------
    # Get all unique coordinate values
    unique_coords = get_unique_coords(ds)
    print(unique_coords['location_id'])  # See all available location IDs
    
    # Get specific coordinates
    unique_coords = get_unique_coords(ds, ['location_id', 'year_id'])
    """
    if coord_names is None:
        # Auto-detect ID coordinates (coordinates ending with '_id')
        coord_names = [coord for coord in ds.coords if coord.endswith('_id')]
    
    result = {}
    for coord in coord_names:
        if coord in ds.coords:
            values = ds[coord].values
            # Sort if numeric, otherwise just convert to list
            try:
                result[coord] = sorted(values)
            except TypeError:
                result[coord] = list(values)
    
    return result


def filter_ds_by_range(ds, **coord_ranges):
    """
    Filter xarray Dataset by coordinate ranges.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to filter
    **coord_ranges : keyword arguments
        Each keyword should be a coordinate name with a tuple (min, max) for the range
        
    Returns:
    --------
    xarray.Dataset
        Filtered dataset
    
    Examples:
    ---------
    # Filter by year range and age range
    filtered_ds = filter_xarray_by_range(
        ds, 
        year_id=(2020, 2022),
        age_group_id=(5, 15)
    )
    """
    # Validate coordinates
    for coord_name in coord_ranges.keys():
        if coord_name not in ds.coords:
            raise ValueError(f"Coordinate '{coord_name}' not found in dataset")
    
    if not coord_ranges:
        return ds
    
    # Build range conditions
    conditions = []
    for coord_name, (min_val, max_val) in coord_ranges.items():
        coord_values = ds[coord_name]
        range_condition = (coord_values >= min_val) & (coord_values <= max_val)
        conditions.append(range_condition)
    
    # Combine all conditions
    combined_mask = reduce(operator.and_, conditions)
    
    return ds.where(combined_mask, drop=True)