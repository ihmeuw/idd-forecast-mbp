from pathlib import Path

MODEL_ROOT = Path("/mnt/team/idd/pub/forecast-mbp")

REPO_ROOT = Path("/mnt/share/homes/bcreiner/repos")

RAW_DATA_PATH = MODEL_ROOT / "01-raw_data"
PROCESSED_DATA_PATH = MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = MODEL_ROOT / "03-modeling_data"
FORECASTING_DATA_PATH = MODEL_ROOT / "04-forecasting_data"
UPLOAD_DATA_PATH = MODEL_ROOT / "05-upload_data"
VISUALIZATION_PATH = MODEL_ROOT / "06-visualization"
FIGURES_PATH = MODEL_ROOT / "07-figures"
GBD_DATA_PATH = f"{RAW_DATA_PATH}/gbd"
LSAE_INPUT_PATH = PROCESSED_DATA_PATH / "lsae_1209"

FHS_RESULTS_PATH = '/mnt/share/forecasting/data/9/future'

repo_name = "idd-forecast-mbp"
package_name = "idd_forecast_mbp"


# Constants
years = list(range(1970, 2101))
past_years = list(range(1970, 2024))
model_years = list(range(2000, 2101))
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


aa_merge_variables = ["location_id", "year_id"]
as_merge_variables = ["location_id", "year_id", "age_group_id", "sex_id"]
#
draws = [f"{i:03d}" for i in range(100)]
fhs_draws = [f"draw_{i}" for i in range(100)]

# Maps for various constants
cause_map = {
    'malaria':{
        'cause_id': 345,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Malaria',
        'fhs_cause_name': 'malaria'
    },
    'dengue': {
        'cause_id': 357,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Dengue',
        'fhs_cause_name': 'ntd_dengue'
    }
}

malaria_variables = {
    "pfpr": f"{LSAE_INPUT_PATH}/malaria_pfpr_mean_cc_insensitive.parquet",
    "incidence": f"{LSAE_INPUT_PATH}/malaria_pf_inc_rate_mean_cc_insensitive.parquet",
    "mortality": f"{LSAE_INPUT_PATH}/malaria_pf_mort_rate_mean_cc_insensitive.parquet",
}
dengue_variables = {
    "dengue_suitability": f"{LSAE_INPUT_PATH}/dengue_suitability_mean_cc_insensitive.parquet"
}

modeling_measure_map = {
    "malaria": {
        "mortality": {
            "short": "malaria_mort_rate",
            "gbd_measure_id": 1,
            "gbd_metric_id": 3,
            "count_name": 'malaria_mort_count',
            "transformation": "log",
        },
        "incidence": {
            "short": "malaria_inc_rate",
            "gbd_measure_id": 6,
            "gbd_metric_id": 3,
            "count_name": 'malaria_inc_count',
            "transformation": "log"
        }
    },
    "dengue": {
        "incidence": {
            "short": "dengue_inc_rate",
            "gbd_measure_id": 6,
            "gbd_metric_id": 3,
            "count_name": 'dengue_inc_count',
            "transformation": "log"
        },
        "cfr": {
            "short": "dengue_cfr",
            "gbd_measure_id": [1,6],
            "gbd_metric_id": [3,3],
            "count_name": None,
            "transformation": "logit"
        }
    }
}

measure_map = {
    "mortality": {
        "measure_id": 1,
        "name": "mortality",
        "rate_name": "Mortality rate",
        "count_name": "Deaths",
        "short": "mort",
    },
    "incidence": {
        "measure_id": 6, 
        "name": "incidence",
        "rate_name": "Incidence rate",
        "count_name": "Cases",
        "short": "inc",
    }
}


ds_coords = ['location_id', 'year_id', 'sex_id', 'age_group_id']

fhs_population_paths = {
    'ssp245': f'{FHS_RESULTS_PATH}/population/20250709_first_sub_rcp45_climate_ref_100d_hiv_shocks_covid_all/population_agg.nc',
    'ssp126': f'{FHS_RESULTS_PATH}/population/20250709_first_sub_rcp26_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all/population_agg.nc',
    'ssp585': f'{FHS_RESULTS_PATH}/population/20250709_first_sub_rcp85_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all/population_agg.nc',
}


full_measure_map = {
    "mortality": {
        "measure_id": 1,
        "name": "mortality",
        "rate_name": "Mortality rate",
        "count_name": "Deaths",
        "short": "mort",
        'fhs_name': 'death',
        'ssp126': {
            'rate': '20250709_first_sub_rcp26_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all_s8',
            'count': '20250709_first_sub_rcp26_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all_s8_num'
        },
        'ssp245': {
            'rate': '20250709_first_sub_rcp45_climate_ref_100d_hiv_shocks_covid_all_s8',
            'count': '20250709_first_sub_rcp45_climate_ref_100d_hiv_shocks_covid_all_s8_num'
        },
        'ssp585': {
            'rate': '20250709_first_sub_rcp85_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all_s8',
            'count': '20250709_first_sub_rcp85_first_sub_climate_vector_borne_diseases_100d_hiv_shocks_covid_all_s8_num'
        }
    },
    "incidence": {
        "measure_id": 6, 
        "name": "incidence",
        "rate_name": "Incidence rate",
        "count_name": "Cases",
        "short": "inc",
        'fhs_name': 'incidence',
        'ssp126': {
            'rate': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg',
            'count': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg_num'
        },
        'ssp245': {
            'rate': '20250719_rcp45_first_sub_climate_ref_scen0_agg',
            'count': '20250719_rcp45_first_sub_climate_ref_scen0_agg_num'
        },
        'ssp585': {
            'rate': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg',
            'count': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg_num'
        }
    },
    "daly": {
        "measure_id": 2, 
        "name": "daly",
        "rate_name": "DALY rate",
        "count_name": "DALYs",
        "short": "daly",
        'fhs_name': 'daly',
        'ssp126': {
            'rate': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg',
            'count': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg_num'
        },
        'ssp245': {
            'rate': '20250719_rcp45_first_sub_climate_ref_agg',
            'count': '20250719_rcp45_first_sub_climate_ref_agg_num'
        },
        'ssp585': {
            'rate': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg',
            'count': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg_num'
        }
    },
    "yld": {
        "measure_id": 3, 
        "name": "yld",
        "rate_name": "YLD rate",
        "count_name": "YLDs",
        "short": "yld",
        'fhs_name': 'yld',
        'ssp126': {
            'rate': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg',
            'count': '20250719_rcp26_first_sub_climate_vector_borne_diseases_scen75_agg_num'
        },
        'ssp245': {
            'rate': '20250719_rcp45_first_sub_climate_ref_scen0_agg',
            'count': '20250719_rcp45_first_sub_climate_ref_scen0_agg_num'
        },
        'ssp585': {
            'rate': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg',
            'count': '20250719_rcp85_first_sub_climate_vector_borne_diseases_scen76_agg_num'
        }
    },
    "yll": {
        "measure_id": 4, 
        "name": "yll",
        "rate_name": "YLL rate",
        "count_name": "YLLs",
        "short": "yll",
        'fhs_name': 'yll',
        'ssp126': {
            'rate' : '20250709_rcp26_first_sub_climate_vector_borne_diseases_agg',
            'count': '20250709_rcp26_first_sub_climate_vector_borne_diseases_agg_num'
        },
        'ssp245': {
            'rate' : '20250709_rcp45_first_sub_climate_ref_agg',
            'count': '20250709_rcp45_first_sub_climate_ref_agg_num'
        },
        'ssp585': {
            'rate' : '20250709_rcp85_first_sub_climate_vector_borne_diseases_agg',
            'count': '20250709_rcp85_first_sub_climate_vector_borne_diseases_agg_num'
        }
    },
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

ssp_scenario_map = {
    "ssp126": {
        "name": "RCP 2.6",
        "rcp_scenario": 2.6,
        "color": "#046C9A",
        "dhs_scenario": 66,
        "dhs_vbd_scenario": 75
    },
    "ssp245": {
        "name": "RCP 4.5",
        "rcp_scenario": 4.5,
        "color": "#E58601",
        "dhs_scenario": 0,
        "dhs_vbd_scenario": 0
    },
    "ssp585": {
        "name": "RCP 8.5",
        "rcp_scenario": 8.5,
        "color": "#A42820",
        "dhs_scenario": 54,
        "dhs_vbd_scenario": 76
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


problematic_rule_map = {
    'malaria': {
        'incidence': {
            'count_raking_factor_max': 100,        	# Flag if raking factor
            'rate_max': {
                4: 1,
                5: 1
            },
            'count_raking_factor_conditional': 10, 	# Combined with rate condition below
            'rate_max_conditional': 0.2          	# Flag if raking factor > 10 AND rate > 0.2
        },
        'mortality': {
            'count_raking_factor_max': 10000000,        		# Flag if raking factor > 100
            'rate_max': {
                4: 1,
                5: 1
            },					    	# Flag if the rate > 1
            'count_raking_factor_conditional': 10000000,	# This combined with 1 means this is turned off
            'rate_max_conditional': 1         	    	# Flag if raking factor > 10 AND rate > 0.2
        }
    },
    'dengue': {
        'incidence': {
            'count_raking_factor_max': 100000,        	# Flag if raking factor > 100
            'rate_max': {
                4: 1/3,
                5: 1/3
            },
            'count_raking_factor_conditional': 100000, # This combined with 0 means this is turned off
            'rate_max_conditional': 1         	    # Flag if raking factor > 10 AND rate > 0.2
        },
        'mortality': {
            'count_raking_factor_max': 100000,        	# Flag if raking factor > 100
            'rate_max': {
                4: 0.0003,
                5: 0.0003
            },
            'count_raking_factor_conditional': 100000, # This combined with 0 means this is turned off
            'rate_max_conditional': .1        	        # Flag if raking factor > 10 AND rate > 0.2
        }        
    }
}
