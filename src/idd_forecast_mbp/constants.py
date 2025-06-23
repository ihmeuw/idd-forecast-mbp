from pathlib import Path

MODEL_ROOT = Path("/mnt/team/idd/pub/forecast-mbp")

REPO_ROOT = Path("/mnt/share/homes/bcreiner/repos")

RAW_DATA_PATH = MODEL_ROOT / "01-raw_data"
PROCESSED_DATA_PATH = MODEL_ROOT / "02-processed_data"
GBD_DATA_PATH = f"{RAW_DATA_PATH}/gbd"
LSAE_INPUT_PATH = PROCESSED_DATA_PATH / "lsae_1209"


repo_name = "idd-forecast-mbp"
package_name = "idd_forecast_mbp"


# Constants
years = list(range(1970, 2101))
past_years = list(range(1970, 2024))
future_years = list(range(2024, 2101))

# GBD Constants
release_2021_id = 9
release_2023_id = 16
como_2023_v = 1591
codcorrec_2023t_v = 461
dalynator_2023_v = 96
burdenator_2023_v = 360
compare_2023_v = 8234
ages = 22
sexes = 3
dengue_id = 357
malaria_ids = 345

#
draws = [f"{i:03d}" for i in range(100)]
fhs_draws = [f"draw_{i}" for i in range(100)]

# Maps for various constants
cause_map = {
    "malaria":{
        "cause_id": 345
    },
    "dengue": {
        "cause_id": 357
    }
}

malaria_variables = {
    "pfpr": f"{LSAE_INPUT_PATH}/malaria_pfpr_mean_cc_insensitive.parquet",
    "incidence": f"{LSAE_INPUT_PATH}/malaria_pf_inc_rate_mean_cc_insensitive.parquet",
    "mortality": f"{LSAE_INPUT_PATH}/malaria_pf_mort_rate_mean_cc_insensitive.parquet",
}
dengue_variables = {
    "dengue_suit": f"{LSAE_INPUT_PATH}/dengue_suitability_mean_cc_insensitive.parquet"
}


measure_map = {
    "mortality": {
        "measure_id": 1,
        "name": "mortality",
        "rate_name": "Mortality rate",
        "count_name": "Deaths",
        "short": "mort"
    },
    "incidence": {
        "measure_id": 6, 
        "name": "incidence",
        "rate_name": "Incidence rate",
        "count_name": "Cases",
        "short": "inc"
    }
}

metric_map = {
    "rate": {
        "name": "rate",
        "metric_id": 3
    },
    "count": {
        "name": "count",
        "metric_id": 1
    },
}

age_type_map = {
    "all_age": {
        "name": "All Age",
        "age_type": "aa"
    },
    "age_specific": {
        "name": "Age-specific",
        "age_type": "as"
    }
}

ssp_scenarios = {
    "ssp126": {
        "name": "RCP 2.6",
        "rcp_scenario": 2.6,
        "color": "#046C9A",
        "dhs_scenario": 66
    },
    "ssp245": {
        "name": "RCP 4.5",
        "rcp_scenario": 4.5,
        "color": "#E58601",
        "dhs_scenario": 0
    },
    "ssp585": {
        "name": "RCP 8.5",
        "rcp_scenario": 8.5,
        "color": "#A42820",
        "dhs_scenario": 54
    }
}

dah_scenarios = {
    "Baseline": {
        "name": "Baseline",
        "color": "#000000"
    },
    "Constant": {
        "name": "Constant",
        "color": "#5DADE2"
    },
    "Increasing": {
        "name": "Increasing",
        "color": "#27AE60"
    },
    "Decreasing": {
        "name": "Decreasing",
        "color": "#8E44AD"
    }
}