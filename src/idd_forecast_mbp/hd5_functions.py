import h5py
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional  # Add List, Dict, Any here


def write_hdf(df, filepath, key='df', max_retries=3, validate=True, 
              compression='blosc:zstd', complevel=9, **kwargs):
    """
    Write DataFrame to HDF5 with basic validation and retry logic
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to write
    filepath : str
        Path to write the HDF5 file
    key : str, default='df'
        Key to store the dataframe under in HDF5
    max_retries : int, default=3
        Maximum number of retry attempts
    validate : bool, default=True
        Whether to validate the written file by reading it back
    compression : str, default='blosc:zstd'
        Compression algorithm
    complevel : int, default=9
        Compression level (1-9)
    **kwargs : dict
        Additional arguments passed to to_hdf()
    """
    import os
    import time
    import pandas as pd
    
    # Set up HDF5 parameters
    hdf_kwargs = {
        'format': 'table',
        'complib': compression.split(':')[0] if ':' in compression else compression,
        'complevel': complevel,
        **kwargs
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            print(f"💾 Writing to {filepath} with {compression} compression...")
            
            # Write the file
            df.to_hdf(filepath, key=key, mode='w', **hdf_kwargs)
            
            # Basic validation if requested
            if validate:
                test_df = pd.read_hdf(filepath, key=key)
                
                if len(test_df) != len(df):
                    raise ValueError(f'Row count mismatch: {len(test_df)} vs {len(df)}')
                
                if list(test_df.columns) != list(df.columns):
                    raise ValueError('Column names mismatch')
                
                print(f'✅ Validation passed')
            
            # Set file permissions
            os.chmod(filepath, 0o775)
            
            # Report file size
            file_size = os.path.getsize(filepath)
            print(f'✅ File written successfully: {filepath}')
            print(f'📁 File size: {file_size / (1024**2):.1f} MB')
            
            return True
            
        except Exception as e:
            if "Resource temporarily unavailable" in str(e) or "unable to lock" in str(e):
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    print(f"🔒 File lock failed (attempt {attempt + 1}). Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
            
            # For final attempt or other errors, re-raise
            if attempt == max_retries - 1:
                print(f'❌ Failed to write {filepath} after {max_retries} attempts')
            raise e
    
    return False


def benchmark_hdf_compression(df, filepath_base, compressions=None, sample_size=None):
    """
    Benchmark different compression options for HDF5 files
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to test compression on
    filepath_base : str
        Base filepath (will append compression info)
    compressions : list, optional
        List of compression options to test. Defaults to common options.
    sample_size : int, optional
        Use only a sample of the dataframe for testing
        
    Returns:
    --------
    pandas.DataFrame
        Benchmark results with compression ratios and write times
    """
    import time
    
    if compressions is None:
        compressions = [
            ('blosc:lz4', 5),      # Fast compression
            ('blosc:lz4hc', 9),    # Better compression
            ('blosc:zstd', 9),     # Best overall
            ('zlib', 9),           # Standard
            ('bzip2', 9),          # High compression
        ]
    
    # Use sample if specified
    test_df = df.sample(n=min(sample_size, len(df))) if sample_size else df
    
    results = []
    
    for comp, level in compressions:
        try:
            filepath = f"{filepath_base}_{comp.replace(':', '_')}_level{level}.h5"
            
            start_time = time.time()
            
            # Write with compression
            success = write_hdf(test_df, filepath, 
                              compression=comp, 
                              complevel=level,
                              validate=False,  # Skip validation for benchmarking
                              use_checksum=False)
            
            write_time = time.time() - start_time
            
            if success and os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                
                # Test read time
                start_time = time.time()
                _ = pd.read_hdf(filepath, key='df')
                read_time = time.time() - start_time
                
                results.append({
                    'compression': comp,
                    'level': level,
                    'file_size_mb': file_size / (1024**2),
                    'write_time_sec': write_time,
                    'read_time_sec': read_time,
                    'filepath': filepath
                })
                
                print(f"✅ {comp} (level {level}): {file_size / (1024**2):.1f} MB, "
                      f"write: {write_time:.1f}s, read: {read_time:.1f}s")
            else:
                print(f"❌ Failed: {comp} (level {level})")
                
        except Exception as e:
            print(f"❌ Error with {comp} (level {level}): {e}")
    
    benchmark_df = pd.DataFrame(results)
    
    if len(benchmark_df) > 0:
        # Calculate compression ratios relative to the largest file
        max_size = benchmark_df['file_size_mb'].max()
        benchmark_df['compression_ratio'] = max_size / benchmark_df['file_size_mb']
        
        # Sort by a combined score (smaller file size + faster write time is better)
        benchmark_df['score'] = (benchmark_df['file_size_mb'] / benchmark_df['file_size_mb'].max() +
                                benchmark_df['write_time_sec'] / benchmark_df['write_time_sec'].max())
        benchmark_df = benchmark_df.sort_values('score')
        
        print(f"\n📊 Compression Benchmark Results:")
        print(benchmark_df[['compression', 'level', 'file_size_mb', 'compression_ratio', 
                           'write_time_sec', 'read_time_sec']].round(2))
        
        best = benchmark_df.iloc[0]
        print(f"\n🏆 Best overall: {best['compression']} (level {best['level']})")
        
    return benchmark_df

def create_hdf_structure(file_path: str, metadata_df: pd.DataFrame, 
                        draw_columns: List[str], metadata_columns: List[str]) -> None:
    """
    Create HDF5 file structure with metadata and pre-allocated draw columns.
    
    Parameters:
    -----------
    file_path : str
        Path to HDF5 file to create
    metadata_df : pd.DataFrame
        DataFrame containing metadata columns (location_id, year_id, etc.)
    draw_columns : List[str]
        List of draw column names (e.g., ['draw_0', 'draw_1', ...])
    metadata_columns : List[str]
        List of metadata column names to include
    """
    n_rows = len(metadata_df)
    
    with h5py.File(file_path, 'w') as f:
        # Write metadata columns
        for col in metadata_columns:
            if col in metadata_df.columns:
                data = metadata_df[col].values
                # Handle different data types appropriately
                if data.dtype == 'object':
                    # Convert strings to fixed-length byte strings for HDF5
                    max_len = max(len(str(x)) for x in data) if len(data) > 0 else 10
                    dt = f'S{max_len}'
                    data = data.astype(dt)
                f.create_dataset(col, data=data)
        
        # Pre-allocate draw columns
        for draw_col in draw_columns:
            f.create_dataset(draw_col, shape=(n_rows,), dtype='float64', 
                           fillvalue=0.0, compression='gzip')

def write_draw_column(file_path: str, draw_column: str, values: np.ndarray) -> None:
    """
    Write values to a specific draw column in existing HDF5 file.
    
    Parameters:
    -----------
    file_path : str
        Path to existing HDF5 file
    draw_column : str
        Name of draw column to write to
    values : np.ndarray
        Values to write
    """
    with h5py.File(file_path, 'a') as f:
        if draw_column in f:
            f[draw_column][:] = values
        else:
            raise KeyError(f"Draw column '{draw_column}' not found in HDF5 file")

def read_hdf_metadata(file_path: str, metadata_columns: List[str]) -> pd.DataFrame:
    """
    Read only metadata columns from HDF5 file.
    
    Parameters:
    -----------
    file_path : str
        Path to HDF5 file
    metadata_columns : List[str]
        List of metadata column names to read
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with metadata columns
    """
    data = {}
    with h5py.File(file_path, 'r') as f:
        for col in metadata_columns:
            if col in f:
                data[col] = f[col][:]
                # Convert byte strings back to regular strings if needed
                if data[col].dtype.kind == 'S':
                    data[col] = data[col].astype(str)
    
    return pd.DataFrame(data)

def append_to_hdf_table(file_path: str, new_data: pd.DataFrame, 
                       table_name: str = 'table') -> None:
    """
    Append data to an existing HDF5 table (for incremental data addition).
    
    Parameters:
    -----------
    file_path : str
        Path to HDF5 file
    new_data : pd.DataFrame
        New data to append
    table_name : str
        Name of the table in HDF5 file
    """
    # This uses pandas HDFStore for table operations
    with pd.HDFStore(file_path, mode='a') as store:
        if table_name in store:
            store.append(table_name, new_data, format='table')
        else:
            store.put(table_name, new_data, format='table')

def check_hdf_structure(file_path: str) -> Dict[str, Any]:
    """
    Check the structure of an HDF5 file.
    
    Parameters:
    -----------
    file_path : str
        Path to HDF5 file
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with dataset names, shapes, and dtypes
    """
    structure = {}
    with h5py.File(file_path, 'r') as f:
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                structure[name] = {
                    'shape': obj.shape,
                    'dtype': obj.dtype,
                    'compression': obj.compression
                }
        f.visititems(visit_func)
    return structure
