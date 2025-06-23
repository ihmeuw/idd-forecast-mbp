import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
from affine import Affine # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
from typing import Literal, NamedTuple
import itertools
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import load_yaml_dictionary, parse_yaml_dictionary
import argparse
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


parser = argparse.ArgumentParser(description="Forecast malaria pfpr")

# Define arguments
parser.add_argument("--scenario_number", type=int, required=True, help="scenario number (0, 1, or 2)")
parser.add_argument("--draw", type=str, required=True, help="Draw number (e.g., '001', '002', etc.)")



from malaria_uncentered_forecast_functions import (
    fit_malaria_pfpr_model,
    fit_malaria_outcome_model,
    generate_dah_scenarios,
    visualize_dah_scenarios,
    generate_malaria_forecasts,
    plot_pfpr_pred_by_model_dah
)


# Scenarios
ssp_scenarios = ["ssp126", "ssp245", "ssp585"]
# Draws
draws = [f"{i:03d}" for i in range(100)]
# Hierarchy
lsae_hierarchy = "lsae_1209"
fhs_hierarchy = "fhs_2021"

# MODELING_DATA_PATH 
MODELING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/03-modeling_data"
FORECASTING_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data"

# Population
FHS_population_path = "/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/fhs_2021/population.parquet"
LSAE_population_path = "/mnt/team/rapidresponse/pub/climate-aggregates/2025_03_20/results/lsae_1209/population.parquet"

malaria_df_path = f"{MODELING_DATA_PATH}/malaria_stage_2_modeling_df.parquet"
malaria_forecasting_df_path = "{FORECASTING_DATA_PATH}/forecast_scenario_{ssp_scenario}_draw_{draw}_df.parquet"

# Hierarchy path
HIERARCHY_PATH = "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_{hierarchy}.parquet"
hierarchy_df_path = f"{FORECASTING_DATA_PATH}/hierarchy_{lsae_hierarchy}_full.parquet"



forecast_df_path = "{FORECASTING_DATA_PATH}/forecasts_{draw}_{model_name}_{dah_scenario}_{ssp_scenario}.parquet"







hierarchy_df = pd.read_parquet(hierarchy_df_path)
# Population
lsae_population_df = pd.read_parquet(LSAE_population_path)
# Rename population to population_total
lsae_population_df.rename(columns={'population': 'population_total'}, inplace=True)
fhs_population_df = pd.read_parquet(FHS_population_path)
# Rename population to population_total
fhs_population_df.rename(columns={'population': 'population_total'}, inplace=True)






# Read the parquet file
malaria_df = pd.read_parquet(malaria_df_path)

# Define models
pfpr_formula = {
    'uncentered': "logit_malaria_pfpr ~ malaria_suitability + log_gdppc_mean + mal_DAH_total_per_capita + total_precipitation + people_flood_days_per_capita + C(A0_af)",
}

model_path = "{MODELING_DATA_PATH}/malaria_model_{name}.pkl"

pfpr_model = {
    name: fit_malaria_pfpr_model(
        malaria_df,
        formula,
        model_path.format(MODELING_DATA_PATH=MODELING_DATA_PATH, name=name)
    )
    for name, formula in pfpr_formula.items()
}
# For later