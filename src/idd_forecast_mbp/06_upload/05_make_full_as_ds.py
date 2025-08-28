import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from typing import Literal, NamedTuple
import itertools
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.hd5_functions import write_hdf
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids
from idd_forecast_mbp.xarray_functions import convert_to_xarray, write_netcdf, ensure_id_coordinates_are_integers, read_netcdf_with_integer_ids
import os

best_run_date = "2025_08_11"

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = rfc.MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = rfc.MODEL_ROOT / "05-upload_data" / "upload_folders"

full_ds_path_template = "{UPLOAD_DATA_PATH}/{best_run_date}/full_as_ds_{dah_scenario}.nc"
#
measure_map = rfc.measure_map
metric_map = rfc.metric_map
cause_map = rfc.cause_map
ssp_scenarios = rfc.ssp_scenarios
as_merge_variables = rfc.as_merge_variables

hierarchy_ds_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.nc"
hierarchy_ds = read_netcdf_with_integer_ids(hierarchy_ds_path, engine='netcdf4')

# Read the NetCDF versions instead of Parquet
as_full_malaria_ds_path = PROCESSED_DATA_PATH / "as_2003_full_malaria_ds.nc"
as_full_dengue_ds_path = PROCESSED_DATA_PATH / "as_2003_full_dengue_ds.nc"

as_full_malaria_ds = read_netcdf_with_integer_ids(as_full_malaria_ds_path, engine='netcdf4')
as_full_dengue_ds = read_netcdf_with_integer_ids(as_full_dengue_ds_path, engine='netcdf4')

as_path_templates = {
    'malaria': {
        'past': as_full_malaria_ds,
        'future': "{UPLOAD_DATA_PATH}/{run_date}/as_cause_malaria_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}/mean.nc"
    },
    'dengue': {
        'past': as_full_dengue_ds,
        'future': "{UPLOAD_DATA_PATH}/{run_date}/as_cause_dengue_measure_{measure}_metric_{metric}_ssp_scenario_{ssp_scenario}/mean.nc"
    },
}

as_full_population_ds_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population_ds.nc"
as_full_population_ds = read_netcdf_with_integer_ids(as_full_population_ds_path)

# for dah_scenario in ['Baseline', 'Constant']:
for dah_scenario in ['Baseline']:
    causes = list(cause_map.keys())
    measures = list(measure_map.keys())
    ssp_scenarios_list = list(ssp_scenarios)

    datasets = []
    for cause in causes:
        # Use xarray Dataset directly for past data
        past_ds = as_path_templates[cause]['past']
        cause_datasets = []
        for ssp_scenario in ssp_scenarios_list:
            measure_datasets = []
            for measure in measures:
                # Load future data
                as_path = as_path_templates[cause]['future'].format(
                    UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
                    run_date=best_run_date,
                    measure=measure,
                    metric="count",
                    ssp_scenario=ssp_scenario,
                    dah_scenario=dah_scenario
                )
                future_ds = read_netcdf_with_integer_ids(as_path, engine='netcdf4')
                future_ds = future_ds.rename({'mean_value': 'count'})

                # Select past data for this measure and metric
                short_name = f'{cause}_{measure_map[measure]["short"]}_count'
                # Select only years before 2022
                measure_past_ds = past_ds[['location_id', 'year_id', short_name]].to_dataframe().reset_index()
                measure_past_ds = measure_past_ds[measure_past_ds['year_id'] < 2022].copy()
                measure_past_ds = measure_past_ds.rename(columns={short_name: 'count'})
                measure_past_ds = convert_to_xarray(
                    measure_past_ds,
                    dimensions=['location_id', 'year_id'],
                    dimension_dtypes={'location_id': 'int32', 'year_id': 'int16'},
                    variable_dtypes={'count': 'float32'},
                    auto_optimize_dtypes=False
                )

                # Combine past and future using xarray concat
                combined_ds = xr.concat([measure_past_ds, future_ds], dim='year_id')
                combined_ds = combined_ds.sortby('year_id')
                combined_ds = combined_ds.assign_coords(measure=measure)
                combined_ds = combined_ds.expand_dims('measure')
                measure_datasets.append(combined_ds)
            # Combine measures
            ssp_ds = xr.concat(measure_datasets, dim='measure')
            ssp_ds = ssp_ds.assign_coords(ssp_scenario=ssp_scenario)
            ssp_ds = ssp_ds.expand_dims('ssp_scenario')
            cause_datasets.append(ssp_ds)
        # Combine SSP scenarios
        cause_ds = xr.concat(cause_datasets, dim='ssp_scenario')
        cause_ds = cause_ds.assign_coords(cause=cause)
        cause_ds = cause_ds.expand_dims('cause')
        datasets.append(cause_ds)

    full_ds = xr.concat(datasets, dim='cause')
    full_ds = xr.merge([full_ds, as_full_population_ds], join='left')
    full_ds['rate'] = full_ds['count'] / full_ds['population']
    full_ds = xr.merge([full_ds, hierarchy_ds], join='left')
    full_ds = ensure_id_coordinates_are_integers(full_ds)
    #
    full_df_path = full_ds_path_template.format(
        UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
        best_run_date=best_run_date,
        dah_scenario=dah_scenario)
    #
    write_netcdf(
        full_ds,
        full_df_path,
        compression=True,
        compression_level=4,
        chunking=True,
        chunk_by_dim={
            'location_id': 1500,
            'year_id': 79,
            'cause': len(causes),
            'measure': len(measures),
            'ssp_scenario': len(ssp_scenarios_list)
        },
        engine='netcdf4'
    )