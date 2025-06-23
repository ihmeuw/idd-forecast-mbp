import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import xarray as xr # type: ignore
from pathlib import Path
from affine import Affine # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
from shapely import MultiPolygon, Polygon # type: ignore
from typing import Literal, NamedTuple
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import load_yaml_dictionary, parse_yaml_dictionary
import argparse

parser = argparse.ArgumentParser(description="Forecast dengue incidence and CFR for a specific scenario and draw")

# Define arguments
parser.add_argument("--ssp_scenario", type=str, required=True, help="ssp_scenario (e.g., 'ssp126', 'ssp245', 'ssp585')")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")

# Parse arguments
args = parser.parse_args()

ssp_scenario = args.ssp_scenario
draw = args.draw

# MODELING_DATA_PATH 
MODELING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
# FORECASTING_DATA_PATH
FORECASTING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"

dengue_df_path = f"{MODELING_DATA_PATH}/dengue_stage_2_modeling_df.parquet"

input_forecast_df_path = f"{FORECASTING_DATA_PATH}/dengue_forecast_scenario_{ssp_scenario}_draw_{draw}.parquet"
output_forecast_df_path = f"{FORECASTING_DATA_PATH}/dengue_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.parquet"

inc_outcome = 'log_dengue_inc_rate'
cfr_outcome = 'logit_dengue_cfr'

# Read the modeling data
dengue_df = pd.read_parquet(dengue_df_path)
dengue_df_location_ids = dengue_df[dengue_df['year_id'] == 2022]['location_id']
dengue_df = dengue_df[dengue_df["location_id"].isin(dengue_df_location_ids)]

def fit_regression_model(data, formula, save_path=None):
    """
    Fit a regression model with given formula and optionally save it
    
    Args:
        data: pandas DataFrame containing model data
        formula: str, model formula
        save_path: str, optional path to save the fitted model
    """
    model = smf.ols(formula, data=data).fit()
    
    if save_path:
        # Save using pickle (recommended method)
        model.save(save_path)
    
    return model

# Define formulas directly
inc_formula = "log_dengue_inc_rate ~ dengue_suitability + total_precipitation + logit_urban_1km_threshold_300 + people_flood_days_per_capita + C(A0_af)"
cfr_formula = "logit_dengue_cfr ~ log_gdppc_mean + logit_urban_1km_threshold_300 + mean_high_temperature + C(A0_af)"


# Fit models directly
inc_model = fit_regression_model(
    dengue_df, 
    inc_formula, 
    f"{MODELING_DATA_PATH}/log_dengue_inc_rate_pred.pkl"
)

cfr_model = fit_regression_model(
    dengue_df, 
    cfr_formula, 
    f"{MODELING_DATA_PATH}/logit_dengue_cfr_pred.pkl"
)

# Load forecast data
forecast_df = pd.read_parquet(
    input_forecast_df_path.format(
        FORECASTING_DATA_PATH=FORECASTING_DATA_PATH, 
        ssp_scenario=ssp_scenario, 
        draw=draw
    )
)

forecast_df = forecast_df[forecast_df['year_id'] >= 2000]
# Create a new column called "in_model" where
# If forecast_df['location_id'] is in dengue_df['location_id'], then set "in_model" to True, else False
forecast_df['in_model'] = forecast_df['location_id'].isin(dengue_df_location_ids)

# Subset to locations that are in the dengue_df in 2022
forecast_df = forecast_df[forecast_df['location_id'].isin(dengue_df_location_ids)]

# Create the A0_af factor variable
forecast_df["A0_location_id"] = forecast_df["A0_location_id"].astype(int)
forecast_df['A0_af'] = 'A0_' + forecast_df['A0_location_id'].astype(str)
forecast_df['A0_af'] = forecast_df['A0_af'].astype('category')
forecast_df['A0_af'] = forecast_df['A0_af'].cat.set_categories(dengue_df['A0_af'].cat.categories)

# Get raw incidence prediction
raw_inc_pred = inc_model.predict(forecast_df)

# Compute incidence shift and apply directly
ref_year_mask = forecast_df['year_id'] == 2022
inc_shift_lookup = dict(zip(
    forecast_df.loc[ref_year_mask, 'location_id'], 
    forecast_df.loc[ref_year_mask, inc_outcome] - raw_inc_pred[ref_year_mask]
))

forecast_df['log_dengue_inc_rate_pred'] = raw_inc_pred + forecast_df['location_id'].map(inc_shift_lookup)

# COMPUTE CFR PREDICTIONS WITH SHIFTING
print("Computing CFR predictions...")

# Update incidence values for CFR prediction
forecast_df[f'inc_outcome_original'] = forecast_df[inc_outcome]
forecast_df[inc_outcome] = forecast_df['log_dengue_inc_rate_pred']

# Get raw CFR prediction
raw_cfr_pred = cfr_model.predict(forecast_df)

# Restore original incidence values
forecast_df[inc_outcome] = forecast_df[f'inc_outcome_original']
forecast_df.drop(columns=[f'inc_outcome_original'], inplace=True)

# Compute CFR shift and apply directly
cfr_shift_lookup = dict(zip(
    forecast_df.loc[ref_year_mask, 'location_id'], 
    forecast_df.loc[ref_year_mask, cfr_outcome] - raw_cfr_pred[ref_year_mask]
))

forecast_df['logit_dengue_cfr_pred'] = raw_cfr_pred + forecast_df['location_id'].map(cfr_shift_lookup)

# Save the updated forecast data
forecast_df.to_parquet(output_forecast_df_path)