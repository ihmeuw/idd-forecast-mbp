import rasterio as rio
import numpy as np
import xarray as xr
import os
from pathlib import Path
from datetime import datetime
from rra_tools.shell_tools import mkdir  # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.yaml_functions import load_yaml_dictionary, parse_yaml_dictionary

repo_name = rfc.repo_name
package_name = rfc.package_name

# Script directory
YAML_PATH = rfc.REPO_ROOT / repo_name / "src" / package_name / "COVARIATE_DICT.yaml"
OUTPUT_PATH = rfc.MODEL_ROOT / "02-processed_data" / "cc_insensitive"
COVARIATE_DICT = load_yaml_dictionary(YAML_PATH)

# Function to set 775 permissions on a file
def set_file_permissions(file_path):
    """Set 775 permissions (rwxrwxr-x) on the specified file."""
    try:
        os.chmod(file_path, 0o775)  # 0o775 is octal for 775 permissions
    except Exception as e:
        print(f"Warning: Could not set permissions on {file_path}: {str(e)}")

def geotiff_to_netcdf(input_path, output_path, variable_name, resolution=None, year=None, encoding=None,
                      global_extent=True, fill_value=0.0):
    """
    Convert a GeoTIFF file to netCDF format.
    
    Parameters:
    -----------
    input_path : str
        Path to the input GeoTIFF file
    output_path : str
        Path to save the output netCDF file
    variable_name : str
        Name of the variable to use in the netCDF file
    resolution : float, optional
        Resolution of the raster in degrees. If None, will be calculated from the transform.
    year : int, optional
        Year for the data (for time dimension)
    encoding : dict, optional
        Encoding options for xarray.to_netcdf()
    global_extent : bool, default=False
        If True, extend the raster to global lat/lon coverage (-180 to 180, -90 to 90)
    fill_value : float, default=0.0
        Value to fill extended areas with
    
    Returns:
    --------
    xr.Dataset: The created dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the GeoTIFF file
    with rio.open(input_path) as src:
        # Read the data and transform information
        data = src.read(1)  # Read first band
        height, width = data.shape
        
        # Handle no-data values
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        
        # Get geospatial information
        transform = src.transform
        
        # Use provided resolution or calculate it from transform
        if resolution is None:
            x_res = transform[0]  # Width of a pixel
            y_res = abs(transform[4])  # Height of a pixel
        else:
            x_res = y_res = resolution  # Use the square pixels resolution from YAML
        
        # Calculate bounds correctly with full pixel coverage
        xmin = transform[2]  # Left edge
        ymax = transform[5]  # Top edge
        
        # Use resolution to calculate exact right and bottom edges
        xmax = xmin + (width * x_res)  # Right edge (including full pixel width)
        ymin = ymax - (height * y_res)  # Bottom edge (including full pixel height)
        
        if global_extent:
            # Define global extents
            global_xmin, global_xmax = -180.0, 180.0
            global_ymin, global_ymax = -90.0, 90.0
            
            # Calculate number of pixels needed for global coverage
            global_width = int(np.ceil((global_xmax - global_xmin) / x_res))
            global_height = int(np.ceil((global_ymax - global_ymin) / y_res))
            
            # Create empty global array filled with fill_value
            global_data = np.full((global_height, global_width), fill_value, dtype=data.dtype)
            
            # Calculate indices to place original data
            x_start = int(round((xmin - global_xmin) / x_res))
            y_start = int(round((global_ymax - ymax) / y_res))
            
            # Ensure indices are within bounds
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            
            # Place the original data into the global grid
            x_end = min(x_start + width, global_width)
            y_end = min(y_start + height, global_height)
            
            # Calculate how much of the original data we can fit
            orig_x_slice = slice(0, x_end - x_start)
            orig_y_slice = slice(0, y_end - y_start)
            
            # Place the data
            global_data[y_start:y_end, x_start:x_end] = data[orig_y_slice, orig_x_slice]
            
            # Use this data for the rest of the function
            data = global_data
            xmin, xmax = global_xmin, global_xmax
            ymin, ymax = global_ymin, global_ymax
            width, height = global_width, global_height
        
        # Create coordinate arrays for pixel centers (not edges)
        lons = np.linspace(xmin + x_res/2, xmax - x_res/2, width)
        lats = np.linspace(ymax - y_res/2, ymin + y_res/2, height)
        
        # Create xarray dataset
        if year is not None:
            # Create a dataset with time dimension using integer year
            time_val = np.array([int(year)])  # Ensure it's an integer in an array
            ds = xr.Dataset(
                {"value": (["time", "lat", "lon"], data[np.newaxis, :, :])},
                coords={
                    "lon": lons,
                    "lat": lats,
                    "time": time_val
                }
            )
            
            # Add time coordinate attributes indicating it's just a year, not a timestamp
            ds.time.attrs = {
                "long_name": "year",
                "standard_name": "year",
                "units": "year"
            }
        else:
            # Create a dataset without time dimension (synoptic)
            ds = xr.Dataset(
                {"value": (["lat", "lon"], data)},
                coords={
                    "lon": lons,
                    "lat": lats
                }
            )
        
        # Add variable attributes
        ds["value"].attrs = {
            "long_name": variable_name.replace("_", " "),
            "units": src.meta.get("units", "unknown"),
            "resolution": x_res  # Add resolution to metadata
        }
        
        # Add coordinate attributes
        ds.lon.attrs = {
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
            "resolution": x_res
        }
        
        ds.lat.attrs = {
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
            "resolution": y_res
        }
        
        # Add global attributes
        ds.attrs = {
            "title": f"{variable_name} data",
            "source": f"Converted from GeoTIFF: {os.path.basename(input_path)}",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Conventions": "CF-1.8",
            "history": f"Created from {input_path}",
            "pixel_size_degrees": x_res,
            "pixel_registration": "center"
        }
        
        # Use default encoding if none provided
        if encoding is None:
            encoding = {
                "value": {"zlib": True, "complevel": 5, "dtype": "float32"},
                "lon": {"dtype": "float32", "zlib": True, "complevel": 5},
                "lat": {"dtype": "float32", "zlib": True, "complevel": 5}
            }
            if year is not None:
                # Ensure time is stored as a simple integer with no date units
                encoding["time"] = {"dtype": "int32"}
        
        # Save the dataset to netCDF
        ds.to_netcdf(output_path, encoding=encoding)
        
        # Set proper permissions
        set_file_permissions(output_path)
        
        return ds


def create_multiyear_netcdf(input_path_template, years, output_path, variable_name):
    """
    Create a multi-year netCDF file from individual GeoTIFF files.
    
    Parameters:
    -----------
    input_path_template : str
        Template path with {year} to be replaced for each year
    years : list of int
        List of years to process
    output_path : str
        Path to save the combined netCDF file
    variable_name : str
        Name of the variable in the output file
    
    Returns:
    --------
    xr.Dataset: The created dataset with time dimension
    """
    first_file = True
    combined_ds = None
    
    for year in years:
        input_path = input_path_template.format(year=year)
        
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Warning: File {input_path} does not exist. Skipping.")
            continue
        
        # Create a temporary dataset
        temp_path = f"/tmp/{os.path.basename(input_path)}.nc"
        temp_ds = geotiff_to_netcdf(input_path, temp_path, variable_name, year=year)
        
        # For the first valid file, initialize the combined dataset
        if first_file:
            combined_ds = temp_ds
            first_file = False
        else:
            # Concatenate along time dimension
            combined_ds = xr.concat([combined_ds, temp_ds], dim="time")
        
        # Remove temporary file
        os.remove(temp_path)
    
    # Sort by time
    if combined_ds is not None:
        combined_ds = combined_ds.sortby("time")
        
        # Define encoding for the combined file
        encoding = {
            "value": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "lon": {"dtype": "float32", "zlib": True, "complevel": 5},
            "lat": {"dtype": "float32", "zlib": True, "complevel": 5},
            "time": {"dtype": "int32"}
        }

        # Update time attributes to ensure it's treated as simple year integers
        if combined_ds is not None:
            combined_ds.time.attrs = {
                "long_name": "year",
                "standard_name": "year", 
                "units": "year"
            }
        
        # Save the combined dataset
        combined_ds.to_netcdf(output_path, encoding=encoding)
        
        return combined_ds
    else:
        print("No valid data files found.")
        return None


def process_synoptic_variable(covariate_key, output_dir):
    """
    Process a synoptic (time-invariant) variable and convert to netCDF.
    
    Parameters:
    -----------
    covariate_key : str
        Key of the covariate in the COVARIATE_DICT
        
    Returns:
    --------
    str: Path to the output netCDF file
    """
    # Get dictionary from YAML
    covariate_dict = parse_yaml_dictionary(covariate_key)
    covariate_name = covariate_dict.get('covariate_name', covariate_key)
    path = covariate_dict['path']
    resolution = covariate_dict.get('covariate_resolution', None)
    
    # Create output directory if it doesn't exist
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = output_dir / f"{covariate_name}.nc"
    
    # Convert to netCDF with resolution from YAML
    geotiff_to_netcdf(
        input_path=path,
        output_path=str(output_path),
        variable_name=covariate_name,
        resolution=resolution
    )
    
    return str(output_path)
   

def process_multiyear_variable(covariate_key, output_dir):
    """
    Process a multi-year variable and convert to a single netCDF with time dimension.
    
    Parameters:
    -----------
    covariate_key : str
        Key of the covariate in the COVARIATE_DICT
        
    Returns:
    --------
    str: Path to the output netCDF file
    """
    # Get dictionary from YAML
    covariate_dict = parse_yaml_dictionary(covariate_key)
    covariate_name = covariate_dict.get('covariate_name', covariate_key)
    path_template = covariate_dict['path']
    years = covariate_dict['years']
    resolution = covariate_dict.get('covariate_resolution', None)
  
    if not years:
        print(f"No years specified for {covariate_name}. Cannot process as multi-year variable.")
        return None
    
    # Create output directory if it doesn't exist
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = output_dir / f"{covariate_name}.nc"
    
    # Update create_multiyear_netcdf to pass resolution to geotiff_to_netcdf
    # This requires modifying the create_multiyear_netcdf function
    
    # For now, pass resolution directly to each geotiff_to_netcdf call
    first_file = True
    combined_ds = None
    
    for year in years:
        input_path = path_template.format(year=year)
        
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Warning: File {input_path} does not exist. Skipping.")
            continue
        
        # Create a temporary dataset
        temp_path = f"/tmp/{os.path.basename(input_path)}.nc"
        temp_ds = geotiff_to_netcdf(
            input_path=input_path, 
            output_path=temp_path, 
            variable_name=covariate_name, 
            resolution=resolution,
            year=year
        )
        
        # For the first valid file, initialize the combined dataset
        if first_file:
            combined_ds = temp_ds
            first_file = False
        else:
            # Concatenate along time dimension
            combined_ds = xr.concat([combined_ds, temp_ds], dim="time")
        
        # Remove temporary file
        os.remove(temp_path)
    
    if combined_ds is not None:
        # Sort by time
        combined_ds = combined_ds.sortby("time")
        
        # Define encoding for the combined file
        encoding = {
            "value": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "lon": {"dtype": "float32", "zlib": True, "complevel": 5},
            "lat": {"dtype": "float32", "zlib": True, "complevel": 5},
            "time": {"dtype": "int32"}
        }
        
        # Save the combined dataset
        combined_ds.to_netcdf(output_path, encoding=encoding)
        
        # Set proper permissions
        set_file_permissions(output_path)
        
    return str(output_path) if combined_ds is not None else None


def batch_process_all_covariates(output_dir=None, skip_existing=True):
    """
    Process all covariates in the COVARIATE_DICT and convert them to netCDF.
    
    Parameters:
    -----------
    output_dir : str or Path, optional
        Directory to save output files. If None, uses default directory.
    skip_existing : bool, default=True
        If True, skip processing covariates that already have output files.
        
    Returns:
    --------
    dict: Dictionary with covariate names as keys and output paths as values
    """
    if output_dir is None:
        output_dir = OUTPUT_PATH
    else:
        output_dir = Path(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for covariate_key in COVARIATE_DICT.keys():
        print(f"\nProcessing {covariate_key}...")
        covariate_dict = parse_yaml_dictionary(covariate_key)
        covariate_name = covariate_dict['covariate_name']

        output_path = output_dir / f"{covariate_name}.nc"
        
        # Skip if output file already exists and skip_existing is True
        if skip_existing and os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping.")
            results[covariate_name] = str(output_path)
            continue
        
        try:
            synoptic = covariate_dict['synoptic']
            years = covariate_dict['years']
            
            if synoptic or len(years) == 1:
                # Process as synoptic variable
                results[covariate_name] = process_synoptic_variable(covariate_key, output_dir)
            else:
                # Process as multi-year variable
                results[covariate_name] = process_multiyear_variable(covariate_key, output_dir)
        except Exception as e:
            print(f"Error processing {covariate_name}: {str(e)}")
            results[covariate_name] = None
    
    # Display summary
    print("\n==== Processing Summary ====")
    success_count = sum(1 for path in results.values() if path is not None)
    print(f"Successfully processed {success_count} out of {len(results)} covariates")
    
    return results


def main():
    """Main function to run when script is executed directly."""
    results = batch_process_all_covariates(skip_existing=False)

    # Print successful conversions
    print("\nSuccessfully converted covariates:")
    for covariate, path in results.items():
        if path:
            print(f"- {covariate}: {path}")

    # Print failed conversions
    print("\nFailed conversions:")
    for covariate, path in results.items():
        if not path:
            print(f"- {covariate}")


if __name__ == "__main__":
    main()