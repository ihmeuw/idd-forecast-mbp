
import os
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Make my agregations")

# Define arguments
parser.add_argument("--location_id", type=int, required=True, help="location_id")
parser.add_argument("--ssp_scenario", type=str, required=True, help="ssp_scenario")
parser.add_argument("--dah_scenario", type=str, required=False, help="dah_scenario")
parser.add_argument("--cause_id", type=int, required=True, help="cause_id")

UPLOAD_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/05-upload_data"

cause_map = {
    345: "malaria",
    357: "dengue"
}

measure_map = {
    "mort": {
        "rate": "df_mort_rate",
        "count": "df_mort_count"
    },
    "inc": {
        "rate": "df_inc_rate",
        "count": "df_inc_count"
    }
}

cause_draw_results = {
    "malaria":{
        "outcome_all_locations": "{UPLOAD_DATA_PATH}/full_hierarchy_malaria_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.parquet",
        "key": "cause_id_345_measure_id_{measure_id}_metric_id_{metric_id}_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}",
        "cause_id": 345
    },
    "dengue": {
        "outcome_all_locations": "{UPLOAD_DATA_PATH}/full_hierarchy_dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet",
        "key": "cause_id_357_measure_id_{measure_id}_metric_id_{metric_id}_ssp_scenario_{ssp_scenario}",
        "cause_id": 357
    }
}

fhs_draws = [f"draw_{i}" for i in range(100)]
draws = [f"{i:03d}" for i in range(100)]

def get_draw_location_results(cause, ssp_scenario, draw, location_id, dah_scenario = None):
    """
    Get the file path for the draw results of a specific cause, SSP scenario, DAH scenario, draw number, and measure ID.
    """
    
    fhs_draw = f"draw_{int(draw)}"
    if cause == "malaria":
        df_template = cause_draw_results["malaria"]["outcome_all_locations"].format(
            UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
            ssp_scenario=ssp_scenario, dah_scenario=dah_scenario, draw=draw
        )
        cause = "malaria_pf"
    elif cause == "dengue":
        df_template = cause_draw_results["dengue"]["outcome_all_locations"].format(
            UPLOAD_DATA_PATH=UPLOAD_DATA_PATH,
            ssp_scenario=ssp_scenario, draw=draw
        )
    df = pd.read_parquet(df_template)
    df = df[df["location_id"] == location_id].reset_index(drop=True)
    #
    drop_columns = ['population', 'draw', 'location_set_version_id', 'location_set_id', 'is_estimate', 'most_detailed_fhs', 'sort_order', 'map_id', 
                    'ihme_loc_id', 'local_id','most_detailed_lsae']
    df = df.drop(columns=drop_columns, errors='ignore')
    # Rename the population_total column to population
    df = df.rename(columns={"population_total": "population"})
    df[f"{cause}_inc_rate"] = df[f"{cause}_inc_count"] / df["population"]
    df[f"{cause}_mort_rate"] = df[f"{cause}_mort_count"] / df["population"]
    df[f'inc_rate_{fhs_draw}'] = df[f"{cause}_inc_count_pred"] / df["population"]
    df[f'mort_rate_{fhs_draw}'] = df[f"{cause}_mort_count_pred"] / df["population"]
    df = df.rename(columns={f"{cause}_inc_count_pred": f"inc_count_{fhs_draw}"})
    df = df.rename(columns={f"{cause}_mort_count_pred": f"mort_count_{fhs_draw}"})

    # Make the draw columns last
    draw_columns = [col for col in df.columns if "draw" in col]
    outcome_columns = [col for col in df.columns if cause in col]    
    other_columns = [col for col in df.columns if "draw" not in col and cause not in col]
    df = df[other_columns + outcome_columns + draw_columns]

    df = df.rename(columns={f"{cause}_inc_count": f"inc_count"})   
    df = df.rename(columns={f"{cause}_inc_rate": f"inc_rate"})
    df = df.rename(columns={f"{cause}_mort_count": f"mort_count"})
    df = df.rename(columns={f"{cause}_mort_rate": f"mort_rate"})
    # Drop the "population" column and the "is_estimate column
    return df

def get_ad_location_results(cause, ssp_scenario, location_id, dah_scenario = None):
    """
    Get the file path for the results based on the cause and scenario.
    """
    print(f"Getting results for {cause} in {ssp_scenario} for location {location_id} with dah_scenario {dah_scenario}")
    df = get_draw_location_results(cause, ssp_scenario, draws[0], location_id, dah_scenario)
    for ix, draw in enumerate(draws[1:], start=1):
        if ix % 10 == 0:
            print(f"Processing draw {draw} for {cause} in {ssp_scenario} for location {location_id} with dah_scenario {dah_scenario}")
        df_draw = get_draw_location_results(cause, ssp_scenario, draw, location_id, dah_scenario)
        draw_columns = [col for col in df_draw.columns if "draw" in col]
        # Drop all columns that aren't draw columns
        df_draw = df_draw[['year_id']+draw_columns]
        # Merge the draw columns with the existing dataframe
        df = df.merge(df_draw, on='year_id', how='left', suffixes=('', f'_{draw}'))
    return df

def split_df_by_measure_and_metric_and_write(df):
    """
    Split the dataframe by measure_id and metric_id.
    """
    print("Splitting dataframe by measure_id and metric_id")
    inc_rate_cols = [col for col in df.columns if 'inc_rate' in col]
    mort_rate_cols = [col for col in df.columns if 'mort_rate' in col]
    inc_count_cols = [col for col in df.columns if 'inc_count' in col]
    mort_count_cols = [col for col in df.columns if 'mort_count' in col]
    other_cols = [col for col in df.columns if col not in inc_rate_cols and col not in mort_rate_cols and col not in inc_count_cols and col not in mort_count_cols]
    #
    df_inc_rate = df[other_cols + inc_rate_cols].copy()
    df_inc_rate['measure_id'] = 6
    df_inc_rate['metric_id'] = 3
    #
    df_mort_rate = df[other_cols + mort_rate_cols].copy()
    df_mort_rate['measure_id'] = 1
    df_mort_rate['metric_id'] = 3
    #
    df_inc_count = df[other_cols + inc_count_cols].copy()
    df_inc_count['measure_id'] = 6
    df_inc_count['metric_id'] = 1
    #
    df_mort_count = df[other_cols + mort_count_cols].copy()
    df_mort_count['measure_id'] = 1
    df_mort_count['metric_id'] = 1
    #
    dfs = {}
    for measure in measure_map.keys():
         dfs[measure] = {}
         for metric in measure_map[measure].keys():
            if measure == "inc" and metric == "rate":
                dfs[measure][metric] = df_inc_rate
            elif measure == "inc" and metric == "count":
                dfs[measure][metric] = df_inc_count
            elif measure == "mort" and metric == "rate":
                dfs[measure][metric] = df_mort_rate
            elif measure == "mort" and metric == "count":
                dfs[measure][metric] = df_mort_count
    return dfs

def write_to_h5(cause, dfs, location_id, ssp_scenario, dah_scenario = None):
    """
    Write the dataframe to an HDF5 file.
    """
    print(f"Writing results for {cause} in {ssp_scenario} for location {location_id} with dah_scenario {dah_scenario}")
    for measure in measure_map.keys():
         for metric in measure_map[measure].keys():
            df = dfs[measure][metric]
            if df.empty:
                continue
            # Create the key for the HDF5 file
            key = cause_draw_results[cause]["key"].format(
                measure_id=df['measure_id'].iloc[0],
                metric_id=df['metric_id'].iloc[0],
                ssp_scenario=ssp_scenario,
                dah_scenario=dah_scenario
            )

            OUTPUT_DIR = f"{UPLOAD_DATA_PATH}/draw_files/results/{key}"
            # Create the folder if it doesn't exist
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            h5_path = f"{OUTPUT_DIR}/location_id_{location_id}_draws.h5"
            df.to_hdf(h5_path, key='df', mode='w', format='table')

            
def main(location_id, cause_id, ssp_scenario, dah_scenario=None):
    """
    Main function to run the aggregation process.
    """
    cause = cause_map[cause_id]
    # Get the results for the location
    print(f"Processing {cause} for location {location_id} in SSP scenario {ssp_scenario} with DAH scenario {dah_scenario}")
    df = get_ad_location_results(cause, ssp_scenario, location_id, dah_scenario)
    # Split the dataframe by measure_id and metric_id
    dfs = split_df_by_measure_and_metric_and_write(df)
    # Write the dataframes to HDF5 files
    write_to_h5(cause, dfs, location_id, ssp_scenario, dah_scenario)
    print(f"Finished processing {cause} for location {location_id} in SSP scenario {ssp_scenario} with DAH scenario {dah_scenario}")

if __name__ == "__main__":
    args = parser.parse_args()
    location_id = args.location_id
    ssp_scenario = args.ssp_scenario
    dah_scenario = args.dah_scenario
    cause_id = args.cause_id

    # Run the main function
    main(location_id, cause_id, ssp_scenario, dah_scenario)
