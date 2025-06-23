import pandas as pd
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description="Location-specific dengue files processing")

# Define arguments
parser.add_argument("--location_id", type=int, required=True, help="location_id")
parser.add_argument("--ssp_scenario", type=str, required=True, help="location_id")

# Parse arguments
args = parser.parse_args()

location_id = args.location_id
ssp_scenario = args.ssp_scenario

scenario_map = {
    "ssp245" : 0,
    "ssp126": 66,
    "ssp585": 54,
}

UPLOAD_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/05-upload_data"
GBD_DATA_PATH = f"{UPLOAD_DATA_PATH}/age_specific_gbd"

dengue_draw_forecast_path = "{UPLOAD_DATA_PATH}/full_hierarchy_dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet"
dengue_gbd_df_path = f"{GBD_DATA_PATH}/as_dengue_results.parquet"
fhs_hierarchy_df_path = f"{GBD_DATA_PATH}/fhs_hierarchy.parquet"
age_metadata_df_path = f"{GBD_DATA_PATH}/age_metadata.parquet"
    

# Load the data
age_metadata_df = pd.read_parquet(age_metadata_df_path)
age_metadata_df = age_metadata_df[["age_group_id", "age_group_name"]]

fhs_hierarchy_df = pd.read_parquet(fhs_hierarchy_df_path)

#####################
# Constants
year_ids = list(range(2022, 2101))
sex_ids = [1, 2] 
age_group_ids = age_metadata_df['age_group_id'].unique()
measure_ids = [1,6]
metric_id = 3
release_id = 9
# Scenario based on ssp_scenario
if ssp_scenario in scenario_map:
    scenario = scenario_map[ssp_scenario]
cause_id = 357 # 345 = dengue, 357 = dengue
fhs_draws = [f"draw_{i}" for i in range(100)]
draws = [f"{i:03d}" for i in range(100)]

def process_location_data(location_id):
    """
    Process data for a specific location_id.
    This function will create empty dataframes for each cause_id and measure_id combination.
    """

    # Create empty dataframes for each cause_id and measure_id combination
    dataframes = {}

    for measure_id in measure_ids:
        # Calculate total number of rows
        num_rows = (len(age_group_ids) * len(sex_ids) * len(year_ids))
        
        # Create arrays for each column using numpy's repeat and tile functions
        # This is much more memory efficient than itertools.product
        base_data = {
            'measure_id': np.full(num_rows, measure_id, dtype=np.int32),
            'metric_id': np.full(num_rows, metric_id, dtype=np.int32),
            'cause_id': np.full(num_rows, cause_id, dtype=np.int32),
            'location_id': np.full(num_rows, location_id, dtype=np.int32),        
            'release_id': np.full(num_rows, release_id, dtype=np.int32),
            'scenario': np.full(num_rows, scenario, dtype=np.int32),
        }
        
        # Create the varying columns using numpy broadcasting
        # This creates the cartesian product more efficiently
        n_ages = len(age_group_ids)
        n_sexes = len(sex_ids)
        n_years = len(year_ids)
        
        # Create repeating patterns for each dimension
        base_data['age_group_id'] = np.tile(
            np.repeat(age_group_ids, n_sexes * n_years), 1
        )
        base_data['sex_id'] = np.tile(
            np.repeat(sex_ids, n_years), n_ages
        )
        base_data['year_id'] = np.tile(
            year_ids, n_ages * n_sexes 
        )
        
        # Order the columns as:
        column_order = [
            'measure_id', 'metric_id', 'cause_id', 'age_group_id',
            'sex_id', 'location_id', 'year_id', 'release_id', 'scenario']
        base_data = {col: base_data[col] for col in column_order}

        # Create all draw columns at once with NaN values
        draw_data = {
            draw: np.full(num_rows, 0.0, dtype=np.float64) 
            for draw in fhs_draws
        }
        
        # Combine all data
        all_data = {**base_data, **draw_data}
        
        # Create DataFrame all at once
        df = pd.DataFrame(all_data)
        
        # Store the dataframe with a descriptive key
        key = f"cause_id_{cause_id}_measure_id_{measure_id}_scenario_{scenario}"
        dataframes[key] = df

    location_gbd_df = pd.read_parquet(dengue_gbd_df_path, filters=[("location_id", "=", location_id)])

    if location_gbd_df['val'].sum() == 0:
        print(f"Location {location_id} has no data for cause_id {cause_id}.")
        # Write empty dataframes to HDF5 files
        for key, df in dataframes.items():
            h5_path = f"{UPLOAD_DATA_PATH}/draw_files/tmp_files/{key}_location_id_{location_id}_draws.h5"
            df.to_hdf(h5_path, key='df', mode='w', format='table',data_columns=['location_id', 'sex_id', 'age_group_id', 'year_id', 'measure_id', 'cause_id'])
    else:
        # Read input data for first draw
        test_dengue_draw_input_path = dengue_draw_forecast_path.format(
            UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
            ssp_scenario=ssp_scenario,
            draw="000"
        )

        location_forecast_df = pd.read_parquet(test_dengue_draw_input_path, filters=[("location_id", "=", location_id)])
        location_forecast_df = location_forecast_df[location_forecast_df['year_id'] >= 2022]
        if location_forecast_df.empty:
            print(f"Skipping location {location_id} as no data found.")
            for key, df in dataframes.items():
                        h5_path = f"{UPLOAD_DATA_PATH}/draw_files/tmp_files/{key}_location_id_{location_id}_draws.h5"
                        df.to_hdf(h5_path, key='df', mode='w', format='table',data_columns=['location_id', 'sex_id', 'age_group_id', 'year_id', 'measure_id', 'cause_id'])

        elif location_forecast_df['dengue_inc_count_pred'].sum() == 0 or location_forecast_df['dengue_mort_count_pred'].sum() == 0:
            print(f"Location {location_id} has no predictions for cause_id {cause_id}.")
            # Write empty dataframes to HDF5 files
            for key, df in dataframes.items():
                        h5_path = f"{UPLOAD_DATA_PATH}/draw_files/tmp_files/{key}_location_id_{location_id}_draws.h5"
                        df.to_hdf(h5_path, key='df', mode='w', format='table',data_columns=['location_id', 'sex_id', 'age_group_id', 'year_id', 'measure_id', 'cause_id'])
        else:
            print(f"Processing GBD data for location_id: {location_id}")
            # GBD Stuff
            location_gbd_df.rename(columns={'val': 'rate'}, inplace=True)
            location_gbd_df['count'] = location_gbd_df['rate'] * location_gbd_df['population']
            location_aa_gbd_df = location_gbd_df.groupby(['year_id', 'measure_id']).agg({
                'count': 'sum',
                'population': 'sum',
            })
            location_aa_gbd_df.rename(columns={'count': 'count_2022'}, inplace=True)
            # drop all location_aa_gbd_df columns other than measure_id and rate_2022
            location_aa_gbd_df = location_aa_gbd_df[['count_2022']].reset_index()
            # Drop year_id
            location_aa_gbd_df = location_aa_gbd_df.drop(columns=['year_id'])
            location_gbd_df = location_gbd_df.merge(location_aa_gbd_df, on='measure_id', how='left')
            location_gbd_df['percent'] = location_gbd_df['count'] / location_gbd_df['count_2022']
            # set percent to 0 if count is 0
            location_gbd_df.loc[location_gbd_df['count'] == 0, 'percent'] = 0.0

            for draw_ix, draw in enumerate(draws):
                print(".", end='')
                fhs_draw = fhs_draws[draw_ix]
                
                # Read input data
                dengue_draw_input_path = dengue_draw_forecast_path.format(
                    UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
                    ssp_scenario=ssp_scenario,
                    draw=draw
                )

                location_forecast_df = pd.read_parquet(dengue_draw_input_path, filters=[("location_id", "=", location_id)])
                location_forecast_df = location_forecast_df[location_forecast_df['year_id'] >= 2022]
                if location_forecast_df.empty:
                    print(f"Skipping location {location_id} as no data found.")
                    continue
                pred_inc_count = location_forecast_df[location_forecast_df['year_id'] == 2022]['dengue_inc_count_pred']
                pred_mort_count = location_forecast_df[location_forecast_df['year_id'] == 2022]['dengue_mort_count_pred']

                den_inc_raking_factor = location_aa_gbd_df[location_aa_gbd_df['measure_id'] == 6]['count_2022'].values[0] / pred_inc_count
                den_mort_raking_factor = location_aa_gbd_df[location_aa_gbd_df['measure_id'] == 1]['count_2022'].values[0] / pred_mort_count

                location_forecast_df['dengue_incidence_count'] = location_forecast_df['dengue_inc_count_pred'] * den_inc_raking_factor.values[0]
                location_forecast_df['dengue_mortality_count'] = location_forecast_df['dengue_mort_count_pred'] * den_mort_raking_factor.values[0]
                # Split
                location_inc_forecast_df = location_forecast_df[['year_id', 'dengue_incidence_count']]
                location_inc_gbd_df = location_gbd_df[location_gbd_df['measure_id'] == 6][['age_group_id', 'sex_id', 'count', 'rate', 'percent','population']]
                location_mort_forecast_df = location_forecast_df[['year_id', 'dengue_mortality_count']]
                location_mort_gbd_df = location_gbd_df[location_gbd_df['measure_id'] == 1][['age_group_id', 'sex_id', 'count', 'rate', 'percent','population']]
                # cross join
                location_inc_df = location_inc_forecast_df.merge(
                    location_inc_gbd_df, 
                    how='cross', 
                    suffixes=('_forecast', '_gbd')
                )
                location_mort_df = location_mort_forecast_df.merge(
                    location_mort_gbd_df, 
                    how='cross', 
                    suffixes=('_forecast', '_gbd')
                )
                # Muliply
                location_inc_df['as_dengue_incidence_count'] = location_inc_df['dengue_incidence_count'] * location_inc_df['percent']
                location_mort_df['as_dengue_mortality_count'] = location_mort_df['dengue_mortality_count'] * location_mort_df['percent']
                # Divide
                location_inc_df['as_dengue_incidence_rate'] = location_inc_df['as_dengue_incidence_count'] / location_inc_df['population']
                location_mort_df['as_dengue_mortality_rate'] = location_mort_df['as_dengue_mortality_count'] / location_mort_df['population']
                # Prep for merge
                location_inc_df = location_inc_df[['year_id','age_group_id','sex_id', 'as_dengue_incidence_rate']]
                location_mort_df = location_mort_df[['year_id','age_group_id','sex_id', 'as_dengue_mortality_rate']]
                # rename as_dengue_incidence_rate to fhs_draw
                location_inc_df.rename(columns={'as_dengue_incidence_rate': fhs_draw}, inplace=True)
                location_mort_df.rename(columns={'as_dengue_mortality_rate': fhs_draw}, inplace=True)
                # Update HDF5 files - directly assign the calculated values
                # For incidence (measure_id == 6)
                inc_key = f"cause_id_{cause_id}_measure_id_6_scenario_{scenario}"
                # Set index for both dataframes to enable direct assignment
                inc_df_indexed = dataframes[inc_key].set_index(['year_id', 'age_group_id', 'sex_id'])
                location_inc_indexed = location_inc_df.set_index(['year_id', 'age_group_id', 'sex_id'])

                # Update the specific draw column
                inc_df_indexed[fhs_draw] = location_inc_indexed[fhs_draw]
                dataframes[inc_key] = inc_df_indexed.reset_index()

                # For mortality (measure_id == 1)
                mort_key = f"cause_id_{cause_id}_measure_id_1_scenario_{scenario}"
                mort_df_indexed = dataframes[mort_key].set_index(['year_id', 'age_group_id', 'sex_id'])
                location_mort_indexed = location_mort_df.set_index(['year_id', 'age_group_id', 'sex_id'])

                # Update the specific draw column
                mort_df_indexed[fhs_draw] = location_mort_indexed[fhs_draw]
                dataframes[mort_key] = mort_df_indexed.reset_index()

            # After the loop, write the updated dataframes to HDF5 files
            for key, df in dataframes.items():
                h5_path = f"{UPLOAD_DATA_PATH}/draw_files/tmp_files/{key}_location_id_{location_id}_draws.h5"
                df.to_hdf(h5_path, key='df', mode='w', format='table', 
                        data_columns=['location_id', 'sex_id', 'age_group_id', 'year_id', 'measure_id', 'cause_id'])
                print(f"Saved {h5_path}")

process_location_data(location_id)