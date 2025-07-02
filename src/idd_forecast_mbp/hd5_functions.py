import pandas as pd
import numpy as np
import os


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

