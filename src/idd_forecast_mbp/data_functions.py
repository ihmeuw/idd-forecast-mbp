### Functions
best_run_date = "2025_08_11"

import os
import pickle # type: ignore
import geopandas as gpd
import xarray as xr
import rasterio as rio

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.xarray_functions import read_netcdf_with_integer_ids
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids
from idd_forecast_mbp.color_functions import (
    create_change_colormap, 
    create_outcome_colormap, 
    create_diverging_colors,
    get_colors
)
from idd_forecast_mbp.bin_functions import *
from idd_forecast_mbp.covariate_functions import *
from idd_forecast_mbp.counterfactual_functions import *
from scipy.interpolate import RegularGridInterpolator # type: ignore

PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
UPLOAD_DATA_PATH = rfc.UPLOAD_DATA_PATH

ssp_scenario_map = rfc.ssp_scenario_map

# Hierarchy path
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

ADMIN_SHAPEFILE_TEMPLATE = "/snfs1/WORK/11_geospatial/admin_shapefiles/2023_10_30/lbd_standard_admin_{admin_num}_simplified.shp"
DISPUTED_SHAPEFILE_PATH = "/snfs1/WORK/11_geospatial/admin_shapefiles/2023_10_30/lbd_disputed_mask.shp"

POPULATION_01_PATH = "/mnt/team/rapidresponse/pub/population-model/results/2025_02_21/wgs84_0p01/{year}q1.tif"
POPULATION_1_PATH = "/mnt/team/rapidresponse/pub/population-model/results/2025_02_21/wgs84_0p1/{year}q1.tif"

full_ds_path_template = "{UPLOAD_DATA_PATH}/upload_folders/{best_run_date}/full_aa_ds_{dah_scenario}.nc"
suit_ds_path = f"{PROCESSED_DATA_PATH}/suitability_data.nc"
model_covvariate_ds_path = "{UPLOAD_DATA_PATH}/upload_folders/{best_run_date}/cov_ds_{dah_scenario}.nc"
income_ds_path = f"{PROCESSED_DATA_PATH}/income_data.nc"

aa_full_population_ds_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population_ds.nc"
aa_pop_ds = read_netcdf_with_integer_ids(aa_full_population_ds_path)

malaria_suitability_raster_path_template = '/mnt/share/erf/climate_downscale/results/annual/{ssp_scenario}/malaria_suitability/{draw}.nc'
dengue_suitability_raster_path_template = '/mnt/share/erf/climate_downscale/results/annual/{ssp_scenario}/dengue_suitability/{draw}.nc'

DAILY_FLOOD_SCALED_PATH = "/mnt/team/rapidresponse/pub/flooding/output/fldfrc/{ssp_scenario}/{model}/{measure}_{adjustment}_{year}.nc"
yearly_flood_path_template = "/mnt/team/rapidresponse/pub/flooding/results/annual/raw/{ssp_scenario}/{measure}_{adjustment}_sum/{model}.nc"

PAST_STORM_PATH = '/mnt/share/forecasting/data/IBTrACS/Microsoft_code_output/storm_hours/20240424_fix_extreme_year'
past_yearly_storm_path_template = '{PAST_PATH}/{year}_all.nc'

FUTURE_STORMPATH = '/mnt/share/forecasting/data/IBTrACS/20240424_fix_extreme_year_wind_incl_temp_region_global_coast/annual_quants'
future_yearly_storm_path_template = '{FUTURE_PATH}/{ssp_scenario}_{year}_combined.nc'


with open(os.path.join(str(PROCESSED_DATA_PATH), 'location_lists.pkl'), 'rb') as f:
    location_lists = pickle.load(f)



dah_scenario = 'Baseline'
full_ds_path = full_ds_path_template.format(
    UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
    best_run_date=best_run_date,
    dah_scenario=dah_scenario
)
full_ds = read_netcdf_with_integer_ids(full_ds_path)

suit_ds = read_netcdf_with_integer_ids(suit_ds_path)
income_ds = read_netcdf_with_integer_ids(income_ds_path)



def reproject_cov_slice(cov_layer, population_data, transform):
    # Get the grid coordinates for the flood data
    cov_lons = cov_layer.lon.values
    cov_lats = cov_layer.lat.values
    cov_data = cov_layer.values

    # Handle NaN values in cov_data
    cov_data_filled = np.nan_to_num(cov_data, nan=0.0)

    # Set up the target grid for population
    transform_affine = transform['transform']
    pop_shape = population_data.shape
    # Calculate the corner coordinates
    minx, maxx = -180, 180
    miny, maxy = -90, 90

    minx_pixel, miny_pixel = ~transform_affine * (minx, maxy)
    maxx_pixel, maxy_pixel = ~transform_affine * (maxx, miny)

    minx_pixel = int(np.floor(minx_pixel))
    miny_pixel = int(np.floor(miny_pixel))
    maxx_pixel = int(np.ceil(maxx_pixel))
    maxy_pixel = int(np.ceil(maxy_pixel))

    lon_min, lat_max = transform_affine * (minx_pixel, miny_pixel)
    lon_max, lat_min = transform_affine * (maxx_pixel, maxy_pixel)

    # Create the target grid coordinates for the population data
    lon_pop = np.linspace(lon_min, lon_max, pop_shape[1])
    lat_pop = np.linspace(lat_max, lat_min, pop_shape[0])  # Descending order for latitude

    # Create interpolator function
    interpolator = RegularGridInterpolator(
        (cov_lats, cov_lons), 
        cov_data_filled, 
        method='nearest', 
        bounds_error=False, 
        fill_value=0
    )

    # Create a meshgrid of target coordinates for vectorized interpolation
    lon_mesh, lat_mesh = np.meshgrid(lon_pop, lat_pop)
    points = np.column_stack((lat_mesh.flatten(), lon_mesh.flatten()))

    # Perform the interpolation and reshape to match population grid
    yearly_impact_per_capita = interpolator(points).reshape(pop_shape)

    return yearly_impact_per_capita

def load_population_data(year, resolution):
    """Load population data for a given year and resolution."""
    if resolution == "0.01":
        path = POPULATION_01_PATH.format(year=year)
    elif resolution == "0.1":
        path = POPULATION_1_PATH.format(year=year)
    else:
        raise ValueError("Invalid resolution. Choose '0.01' or '0.1'.")
    
    with rio.open(path) as src:
        population_data = src.read(1)
        population_meta = src.meta
    return population_data, population_meta

def load_suitability_raster_data(map_plot_dict):
    cause = map_plot_dict['cause']
    map_type = map_plot_dict['map_type']
    statistic = map_plot_dict['statistic']
    periods = []
    for period_num in range(1, len(map_plot_dict['periods']) + 1):
        period_dict = map_plot_dict[f'period_{period_num}']
        year = range(period_dict['start_year'], period_dict['end_year'] + 1)
        ssp_scenario = period_dict['ssp_scenario']
        if cause == 'malaria':
            path = malaria_suitability_raster_path_template.format(ssp_scenario=ssp_scenario, draw='000')
        else:
            path = dengue_suitability_raster_path_template.format(ssp_scenario=ssp_scenario, draw='000')
        ds = xr.open_dataset(path)
        ds = ds.sel(year=year)
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        ds = getattr(ds, statistic)(dim='year')
        periods.append(ds)
    if map_type == 'outcome':
        plot_data = periods[0]
        
    elif map_type == 'percent_change':
        # Percent change calculation in xarray
        plot_data = (periods[1] - periods[0]) / periods[0] * 100

    elif map_type in ['change', 'scenario_comparison']:
        # Absolute change/difference calculation in xarray
        plot_data = periods[1] - periods[0]

    reference_population, _ = load_population_data(map_plot_dict[f'period_1']['start_year'], resolution="0.1")
    water_mask = np.isnan(reference_population)
    plot_data = plot_data['value']
    plot_data = plot_data.where(~water_mask)
    return plot_data

def load_cov_data(cov_dict, model = 'MIROC6', measure = 'fldfrc', adjustment = 'shifted0.1'):
    cov = cov_dict['cov']
    cause = cov_dict.get('cause', None)
    year = cov_dict['year']
    ssp_scenario = cov_dict.get('ssp_scenario', 'ssp245')

    if cov == 'floods':
        path = yearly_flood_path_template.format(ssp_scenario=ssp_scenario, measure=measure, adjustment=adjustment, model=model)
        ds = xr.open_dataset(path)
        ds = ds.sel(time=int(year))  # Select the data for the specified year
    elif cov == 'storms':
        if year <= 2022:
            path = past_yearly_storm_path_template.format(PAST_PATH=PAST_STORM_PATH, year=year)
            ds = xr.open_dataset(path)
            # Rename coordinate y to lat, x to lon
            ds = ds.rename({'y': 'lat', 'x': 'lon'})
        else:
            path = future_yearly_storm_path_template.format(FUTURE_PATH=FUTURE_STORMPATH, ssp_scenario=ssp_scenario, year=year)
            ds = xr.open_dataset(path)
            ds = ds.sel(quantile='0.5')
            ds = ds.rename({'predictions': 'value'})
            # Remove the extra dimensions (scenario, year) by squeezing them out
            ds = ds.squeeze()

    # Instead of ds = ds['value'], use the actual variable name
    # For example, if the variable is called 'storm_hours':
    # ds = ds['storm_hours']
    # Or if it's the only data variable, you can get it like this:
    data_vars = list(ds.data_vars.keys())
    if len(data_vars) == 1:
        variable_name = data_vars[0]
        ds = ds[variable_name]
    else:
        print(f"Multiple data variables found: {data_vars}")
        # You'll need to specify which one you want

    return ds

def get_raster_data(map_plot_dict):
    measure = map_plot_dict['measure']
    cause = map_plot_dict['cause']
    metric = map_plot_dict['metric']
    statistic = map_plot_dict['statistic']
    resolution = map_plot_dict['resolution']  # This is your actual resolution!
    periods = map_plot_dict['periods']
    map_type = map_plot_dict['map_type']

    if measure == 'suitability':
        plot_data = load_suitability_raster_data(map_plot_dict)
    else:
        periods = []
        for period_num in range(1, len(map_plot_dict['periods']) + 1):
            period_dict = map_plot_dict[f'period_{period_num}']
            start_year = period_dict['start_year']
            end_year = period_dict['end_year']

            reference_year = start_year
            reference_population, _ = load_population_data(reference_year, resolution=resolution)
            period_count = np.zeros_like(reference_population)
            period_rate = np.zeros_like(reference_population)
            period_population = np.zeros_like(reference_population)
            # Create water mask at the correct resolution
            water_mask = np.zeros_like(reference_population, dtype=bool)

            for year in range(start_year, end_year + 1):
                population_data, transform = load_population_data(year, resolution=resolution)  # Use resolution parameter!
                period_population += population_data
                # Build water mask from actual population data
                year_water_mask = np.isnan(population_data)
                water_mask = water_mask | year_water_mask
                
                cov_dict = {'cov': measure, 'year': year, 'ssp_scenario': period_dict.get('ssp_scenario', 'ssp245'), 'cause': cause}
                cov_ds = load_cov_data(cov_dict)
                cov_value = cov_ds.squeeze()
                yearly_impact_per_capita = reproject_cov_slice(cov_value, population_data, transform)
                period_rate += yearly_impact_per_capita
                yearly_impact = yearly_impact_per_capita * population_data
                period_count += yearly_impact

            if statistic == 'mean':
                period_rate = period_rate / (end_year - start_year + 1)
                period_count = period_count / (end_year - start_year + 1)
                period_population = period_population / (end_year - start_year + 1)

            if metric == 'count':
                period_data = period_count
            else:
                period_data = period_rate
            
            # Apply water mask to numpy array
            period_data[water_mask] = np.nan
            periods.append(period_data)

        # Do map type calculations with numpy arrays
        if map_type == 'outcome':
            plot_data = periods[0]
            
        elif map_type == 'percent_change':
            plot_data = (periods[1] - periods[0]) / periods[0] * 100

        elif map_type in ['change', 'scenario_comparison']:
            plot_data = periods[1] - periods[0]

    get_bin_info(map_plot_dict, plot_data)

def read_polygons():
    admin0_polygons = gpd.read_file(ADMIN_SHAPEFILE_TEMPLATE.format(admin_num=0))
    admin0_polygons = admin0_polygons.rename(columns={"loc_id": "location_id"})
    admin1_polygons = gpd.read_file(ADMIN_SHAPEFILE_TEMPLATE.format(admin_num=1))
    admin1_polygons = admin1_polygons.rename(columns={"loc_id": "location_id"})
    admin2_polygons = gpd.read_file(ADMIN_SHAPEFILE_TEMPLATE.format(admin_num=2))
    admin2_polygons = admin2_polygons.rename(columns={"loc_id": "location_id"})
    disputed_polygons = gpd.read_file(DISPUTED_SHAPEFILE_PATH)

    admin1_polygons = admin1_polygons.merge(admin0_polygons[['ADM0_CODE', 'location_id']].rename(columns={'location_id': 'A0_location_id'}),
                                            on='ADM0_CODE', how='left')
    admin2_polygons = admin2_polygons.merge(admin0_polygons[['ADM0_CODE', 'location_id']].rename(columns={'location_id': 'A0_location_id'}),
                                            on='ADM0_CODE', how='left')
    return admin0_polygons, admin1_polygons, admin2_polygons, disputed_polygons


def update_loc_ids(map_plot_dict):
    cause = map_plot_dict['cause']
    if cause is None:
        cause = 'malaria'
    if map_plot_dict['location_type'] == 'endemic':
        map_plot_dict['figure_dict']['inset_label'] = 'Non-endemic' if cause == 'malaria' else 'No local transmission'
        map_plot_dict['map_a0_loc_ids'] = location_lists['endemic']['loc_id_list'][cause]['a0_loc_ids']
        map_plot_dict['map_a1_loc_ids'] = location_lists['endemic']['loc_id_list'][cause]['a1_loc_ids']
        # map_plot_dict['map_a2_loc_ids'] = location_lists['endemic']['loc_id_list'][cause]['a2_loc_ids']
        map_plot_dict['map_a2_loc_ids'] = location_lists['endemic']['loc_id_list'][cause]['full_a2_loc_ids']
    else:
        map_plot_dict['figure_dict']['inset_label'] = None
        map_plot_dict['map_a0_loc_ids'] = location_lists['all']['loc_id_list'][cause]['a0_loc_ids']
        map_plot_dict['map_a1_loc_ids'] = location_lists['all']['loc_id_list'][cause]['a1_loc_ids']
        map_plot_dict['map_a2_loc_ids'] = location_lists['all']['loc_id_list'][cause]['a2_loc_ids']

def get_admin2_data(map_plot_dict):
    cause = map_plot_dict['cause']
    map_type = map_plot_dict['map_type']
    measure = map_plot_dict['measure']
    metric = map_plot_dict['metric']
    outcome_type = map_plot_dict['outcome_type']
    outcome_label = map_plot_dict['outcome_label']
    full_outcome_label = map_plot_dict['full_outcome_label']
    bin_dict = map_plot_dict['bin_dict']

    update_loc_ids(map_plot_dict)

    # Get the final DataFrame with calculations done in xarray
    df = get_outcome_df(map_plot_dict)
    
    # Add hierarchy information
    if 'A0_location_id' not in df.columns:
        plot_data = df.merge(hierarchy_df[['location_id', 'A0_location_id']], on='location_id', how='left')
    else:
        plot_data = df
    
    # Prepare data and colormap based on map type
    data_column = 'val'
    
    if map_type == 'outcome':
        # Outcome map
        create_outcome_colormap(map_plot_dict)
        if cause is None:
            colorbar_label = full_outcome_label
            title = f'{outcome_label} in {map_plot_dict['period_1']['start_year']} ({ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]})'
        else:
            title = f'{cause.capitalize()} {outcome_label} in {map_plot_dict['period_1']['start_year']} ({ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]})'
            colorbar_label = f'{cause.capitalize()} {full_outcome_label}'
            
    elif map_type == 'percent_change':
        # Percent change map
        if outcome_type == 'gdppc_mean':
            create_change_colormap(map_plot_dict, clip_neg=True)
        elif outcome_type == 'dah_pc':
            create_change_colormap(map_plot_dict, clip_pos=True)
        else:
            create_change_colormap(map_plot_dict)
        if cause is None:
            title = f'Percent change in {outcome_label}: {map_plot_dict['period_1']['start_year']} to {map_plot_dict['period_2']['start_year']} - {ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]}'
            colorbar_label = f'Percent change in {full_outcome_label}'
        else:
            title = f'Percent change in {cause} {outcome_label}: {map_plot_dict['period_1']['start_year']} to {map_plot_dict['period_2']['start_year']} - {ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]}'
            colorbar_label = f'Percent change in {cause.capitalize()} {full_outcome_label}'
            
    elif map_type == 'change':
        # Absolute change map
        if outcome_type == 'gdppc_mean':
            create_change_colormap(map_plot_dict, clip_neg=True)
        elif outcome_type == 'dah_pc':
            create_change_colormap(map_plot_dict, clip_pos=True)
        else:
            create_change_colormap(map_plot_dict)
        if cause is None:
            title = f'Change in {outcome_label}: {map_plot_dict['period_1']['start_year']} to {map_plot_dict['period_2']['start_year']} - {ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]}'
            colorbar_label = f'Change in {full_outcome_label}'
        else:
            title = f'Change in {cause} {outcome_label}: {map_plot_dict['period_1']['start_year']} to {map_plot_dict['period_2']['start_year']} - {ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]}'
            colorbar_label = f'Change in {cause.capitalize()} {full_outcome_label}'
            
    elif map_type == 'scenario_comparison':
        # Scenario comparison map
        create_change_colormap(map_plot_dict)
        if cause is None:
            title = f'Difference in {outcome_label}: {ssp_scenario_map[map_plot_dict['period_2']['ssp_scenario']]["name"]} - {ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]} ({map_plot_dict['period_1']['start_year']})'
            colorbar_label = f'Difference in {full_outcome_label} ({ssp_scenario_map[map_plot_dict['period_2']['ssp_scenario']]["name"]} - {ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]})'
        else:
            title = f'Difference in {cause} {outcome_label}: {ssp_scenario_map[map_plot_dict['period_2']['ssp_scenario']]["name"]} - {ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]} ({map_plot_dict['period_1']['start_year']})'
            colorbar_label = f'Difference in {cause.capitalize()} {outcome_label} ({ssp_scenario_map[map_plot_dict['period_2']['ssp_scenario']]["name"]} - {ssp_scenario_map[map_plot_dict['period_1']['ssp_scenario']]["name"]})'

    map_plot_dict['bin_dict']['bin_labels'] = pretty_bin_labels(map_plot_dict)
    data_dict = {
        'plot_data': plot_data,
        'data_column': data_column
    }
    map_plot_dict['data_dict'] = data_dict
    if map_plot_dict['replace_titles']:
        map_plot_dict['legend_dict']['legend_title'] = colorbar_label
        map_plot_dict['figure_dict']['colorbar_label'] = colorbar_label
    if map_plot_dict['figure_dict']['title'] is None:
        map_plot_dict['figure_dict']['title'] = title
    
def get_plot_data(map_plot_dict):
    data_type = map_plot_dict['data_type']
    if data_type == 'raster':
        get_raster_data(map_plot_dict)
    else:
        get_admin2_data(map_plot_dict)


def make_cov_count(outcome_ds):
    years = outcome_ds['year_id'].values
    location_ids = outcome_ds['location_id'].values
    pop_values = aa_pop_ds.sel(year_id=years, location_id=location_ids)['population']
    outcome_ds = outcome_ds * pop_values

def get_outcome_df(map_plot_dict):
    """
    Optimized function that does calculations in xarray and returns final DataFrame
    """
    cause = map_plot_dict['cause']
    statistic = map_plot_dict['statistic']
    map_type = map_plot_dict['map_type']
    
    # Get datasets for each period (keep as xarray)
    periods = []
    for period_num in range(1, len(map_plot_dict['periods']) + 1):
        period_dict = map_plot_dict[f'period_{period_num}']
        years = list(range(period_dict['start_year'], period_dict['end_year'] + 1))
        
        if map_plot_dict['outcome_type'] == 'suitability':
            # Get suitability data
            outcome_ds = suit_ds.sel(cause=map_plot_dict['cause'], location_id=map_plot_dict['map_a2_loc_ids'], 
                                    year_id=years, ssp_scenario=period_dict['ssp_scenario'])[map_plot_dict['outcome_type']].copy()
        elif map_plot_dict['outcome_type'] == 'urbanization':
            # Get urbanization data
            outcome_ds = get_urban_ds(year_ids=years, location_ids=map_plot_dict['map_a2_loc_ids']).load()
            if map_plot_dict['metric'] == 'count':
                outcome_ds = make_cov_count(outcome_ds)
            
        elif map_plot_dict['outcome_type'] == 'dah_pc':
            # Get DAH data
            outcome_ds = get_dah_ds(year_ids=years, dah_scenarios=['Baseline']).load()
            current_location_ids = set(outcome_ds.location_id.values)
            missing_location_ids_set = set(map_plot_dict['map_a2_loc_ids'])
            all_location_ids = sorted(current_location_ids.union(missing_location_ids_set))
            # Reindex with the combined list
            outcome_ds = outcome_ds.reindex(location_id=all_location_ids, fill_value=0)
            if map_plot_dict['metric'] == 'count':
                outcome_ds = make_cov_count(outcome_ds)
        elif map_plot_dict['outcome_type'] == 'flooding_pc':
            # Get flooding data
            outcome_ds = get_flooding_ds(year_ids=years, location_ids=map_plot_dict['map_a2_loc_ids'], 
                                         ssp_scenarios=[period_dict['ssp_scenario']]).load()
            if map_plot_dict['metric'] == 'count':
                outcome_ds = make_cov_count(outcome_ds)
        elif map_plot_dict['outcome_type'] == 'gdppc_mean':
            outcome_ds = get_income_ds(year_ids=years, location_ids=map_plot_dict['map_a2_loc_ids']).load()
            if map_plot_dict['metric'] == 'count':
                outcome_ds = make_cov_count(outcome_ds)
        else:
            metric = map_plot_dict['metric']
            measure=map_plot_dict['measure']
            hold_variable=period_dict.get('hold_variable', None)
            dah_scenario=period_dict.get('dah_scenario', 'Baseline')
            _, aa_ref_ds_folder = get_aa_path(cause=cause, measure=measure, metric='count',
                                            ssp_scenario=period_dict['ssp_scenario'], dah_scenario=dah_scenario,
                                            hold_variable=hold_variable, run_date='2025_08_28')
            aa_ref_ds_path = ds_path_template.format(folder_path=aa_ref_ds_folder, statistic='draws')
            ds = read_netcdf_with_integer_ids(aa_ref_ds_path)
            ds = ds.mean(dim='draw_id')
            ds = ds.sel(year_id=years)
            ds = ds.load()
            outcome_ds = ds.sel(location_id=map_plot_dict['map_a2_loc_ids'])
            # Apply rate scaling in xarray
            if metric == 'rate':
                outcome_ds = outcome_ds / aa_pop_ds.sel(location_id=outcome_ds['location_id'], year_id=outcome_ds['year_id'])['population']
                outcome_ds = outcome_ds * 100000
                
        outcome_ds = getattr(outcome_ds, statistic)(dim='year_id')
        periods.append(outcome_ds)
    
    # Do map type calculations in xarray
    if map_type == 'outcome':
        result_ds = periods[0]
        
    elif map_type == 'percent_change':
        # Percent change calculation in xarray
        result_ds = (periods[1] - periods[0]) / periods[0] * 100

    elif map_type in ['change', 'scenario_comparison']:
        # Absolute change/difference calculation in xarray
        result_ds = periods[1] - periods[0]

    # Convert to DataFrame only at the very end
    if isinstance(result_ds, xr.Dataset):
        if 'value' in result_ds:
            result_ds = result_ds.rename({'value': 'val'})
    else:
        result_ds = result_ds.rename('val')
    df = result_ds.to_dataframe().reset_index()
    
    # Keep only location_id and val columns
    df = df[['location_id', 'val']].copy()
    
    return df