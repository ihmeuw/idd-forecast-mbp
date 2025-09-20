__all__ = [
    "get_location_id",
    "get_data_dict",
    "build_selection_dict",
    "get_fhs_ds_multi"
]


import xarray as xr
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids


PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
FHS_RESULTS_PATH = rfc.FHS_RESULTS_PATH
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)


cause_map = rfc.cause_map
full_measure_map = rfc.full_measure_map
ds_coords = rfc.ds_coords
fhs_population_paths = rfc.fhs_population_paths



def get_location_id(location_name: str) -> int:
    """Get location_id from location_name using hierarchy_df."""
    location_row = hierarchy_df[hierarchy_df['location_name'] == location_name]
    if location_row.empty:
        raise ValueError(f"Location name '{location_name}' not found in hierarchy.")
    return int(location_row['location_id'].values[0])

def get_data_dict(draws = True, cause = 'malaria', measure = 'daly', metric = 'rate', 
                  ssp_scenarios = ['ssp245'], location_ids = 1, year_ids = 2023, 
                  sex_ids = 3, age_group_ids = 22):
    data_dict = {
        'draws': draws,
        'cause': cause,
        'measure': measure,
        'metric': metric,
        'ssp_scenarios': ssp_scenarios,
        'location_id': location_ids,
        'year_id': year_ids,
        'sex_id': sex_ids,
        'age_group_id': age_group_ids
    }
    return data_dict

def build_selection_dict(data_dict):
    """Build selection dictionary from data_dict, excluding 'all' values."""
    selection_dict = {}
    for coord in ds_coords:
        if data_dict[coord] != 'all':
            selection_dict[coord] = data_dict[coord]
    return selection_dict

def get_fhs_ds_multi(data_dict):
    draws = data_dict['draws']
    ssp_scenarios = data_dict.get('ssp_scenarios', ['ssp245'])
    cause = data_dict['cause']
    measure = data_dict['measure']
    metric = data_dict['metric']
    selection_dict = build_selection_dict(data_dict)
    datasets = []

    measure_data = full_measure_map[measure]
    fhs_name = measure_data['fhs_name']

    for ssp_scenario in ssp_scenarios:
        fhs_metric_path = measure_data[ssp_scenario][metric]
        if draws:
            if cause == 'all':
                file_name = '_all.nc'
            else:
                file_name = cause_map[cause]['fhs_cause_name'] + '.nc'
            ds_path = f'{FHS_RESULTS_PATH}/{fhs_name}/{fhs_metric_path}/{file_name}'
            ds = xr.open_dataset(ds_path)
        else:
            file_name = "summary_agg/summary.nc"
            ds_path = f'{FHS_RESULTS_PATH}/{fhs_name}/{fhs_metric_path}/{file_name}'
            ds = xr.open_dataset(ds_path)
            ds = ds.sel(acause = cause_map[cause]['fhs_cause_name'], statistic = 'mean')
        ds = ds.sel(**selection_dict)
        scalar_coords = [coord for coord, values in ds.coords.items() 
                        if values.ndim == 0]  # ndim == 0 means scalar
        ds = ds.drop_vars(scalar_coords)
        ds = ds.expand_dims({'ssp_scenario': [ssp_scenario]})
        ds = ds.isel(scenario=0, drop=True)
        # If there is a variable in ds names value and there isn't one named draws, rename value to draws
        if 'value' in ds.data_vars and 'draws' not in ds.data_vars:
            ds = ds.rename({'value': 'draws'})
        
        datasets.append(ds)
    combined_ds = xr.concat(datasets, dim='ssp_scenario')
    return combined_ds