import pandas as pd # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from rra_flooding.data import FloodingData
from rra_flooding import constants as rfc
from rra_flooding.helper_functions import parse_yaml_dictionary
import argparse
import yaml # type: ignore

parser = argparse.ArgumentParser(description="Run James code")

# Define arguments
parser.add_argument("--variable", type=str, required=True, help="Variable to process")
parser.add_argument("--adjustment_num", type=int, required=True, help="Adjustment number")
parser.add_argument("--hierarchy", type=str, required=True, help="Hierarchy")
parser.add_argument("--scenario", type=str, required=True, help="Scenario")
parser.add_argument("--variant", type=str, default="r1i1p1f1", help="Model variant identifier")


# Parse arguments
args = parser.parse_args()

variable = args.variable
adjustment_num = args.adjustment_num
hierarchy = args.hierarchy
scenario = args.scenario
variant = args.variant

HIERARCHY_MAP = {
    "gbd_2021": [
        "gbd_2021",
        "fhs_2021",
    ],  # GBD pixel hierarchy maps to GBD and FHS locations
    "lsae_1209": ["lsae_1209"],  # LSAE pixel hierarchy maps to LSAE locations
}


variable_dict = parse_yaml_dictionary(variable, adjustment_num)
summary_variable = variable_dict['summary_variable']

def create_mean_results(hierarchy: str, scenario: str, variant: str) -> None:
    root = Path("/mnt/team/rapidresponse/pub/flooding/results/output/") / hierarchy
    models = [
        "ACCESS-CM2", "EC-Earth3", "INM-CM5-0", "MIROC6", 
        "IPSL-CM6A-LR", "NorESM2-MM", "MRI-ESM2-0",
    ]
    # models = ["GFDL-CM4"] # remove becuase not in ssp126

    df_list = []

    for model in models:
        file_name = f"{summary_variable}_{scenario}_{model}_{variant}.parquet"
        file_path = root / file_name
        if not file_path.exists():
            continue 
        
        df = pd.read_parquet(file_path)
        df_list.append(df)

    if not df_list:
        return 

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df["model"] = "mean"

    combined_df = combined_df.groupby(
        ["location_id", "year_id", "model", "scenario", "variant"]
    ).agg({
        "people_flood_days": "mean", 
        "people_flood_days_per_capita": "mean",
        "population": "mean"
    }).reset_index()

    # Save the aggregated mean results
    output_file = root / f"{summary_variable}_{scenario}_mean_{variant}.parquet"
    combined_df.to_parquet(output_file, index=False)
    

subset_hierarchies = HIERARCHY_MAP[hierarchy]
for subset_hierarchy in subset_hierarchies:
    create_mean_results(
        hierarchy=subset_hierarchy,
        scenario=scenario,
        variant=variant,
    )