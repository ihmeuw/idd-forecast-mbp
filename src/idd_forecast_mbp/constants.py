import os
from pathlib import Path

MODEL_ROOT = Path("/mnt/team/idd/pub/forecast-mbp")

REPO_ROOT = Path("/mnt/share/homes/bcreiner/repos")

# Run date: set IDD_RUN_DATE env var to override (e.g. for re-running a prior date).
# Format: YYYYMMDD. Multiple runs same day: set to YYYYMMDD_v2, etc.
RUN_DATE: str = os.environ.get("IDD_RUN_DATE", "20260405")

# ── Stage root directories ────────────────────────────────────────────────────
# These are the node directories that contain dated run subdirs + current/ symlink.
# Do not use these directly for file I/O — use the write/read paths below.
_RAW_STAGE       = MODEL_ROOT / "01-raw_data"
_PROCESSED_STAGE = MODEL_ROOT / "02-processed_data"
_MODELING_STAGE  = MODEL_ROOT / "03-modeling_data"
_FORECASTING_STAGE = MODEL_ROOT / "04-forecasting_data"
_UPLOAD_STAGE    = MODEL_ROOT / "05-upload_data"
_VIZ_STAGE       = MODEL_ROOT / "06-visualization"
_FIGURES_STAGE   = MODEL_ROOT / "07-figures"
_MANUSCRIPT_STAGE  = MODEL_ROOT / "08-manuscript_material"
_PRESENTATION_STAGE = MODEL_ROOT / "10-presentation_material"


def _artifact_write(artifact_root: Path, run_date: str = None) -> Path:
    """Return the write path for an artifact: artifact_root/RUN_DATE."""
    return artifact_root / (run_date or RUN_DATE)


def _artifact_read(artifact_root: Path) -> Path:
    """Return the read path for an artifact via its current/ symlink."""
    return artifact_root / "current"


# ── External / read-only data (not versioned by this pipeline) ────────────────
RAW_DATA_PATH  = _RAW_STAGE          # GBD pulls, raw inputs — never written by pipeline
GBD_DATA_PATH  = RAW_DATA_PATH / "gbd"

# Covariate data produced by the RapidResponse lsae pipeline. Lives flat in
# 02-processed_data/lsae_XXXX/ and is updated externally, not by this pipeline.
LSAE_HIERARCHY = "lsae_1285"
LSAE_INPUT_PATH = _PROCESSED_STAGE / LSAE_HIERARCHY

# Age-specific FHS metadata (written by get_past_as_aa_fhs_outcomes.r, read-only here).
AGE_SPECIFIC_FHS_PATH = _PROCESSED_STAGE / "age_specific_fhs"

# ── 02-processed_data artifact roots ─────────────────────────────────────────
# Each artifact dir contains: {hierarchy}/RUN_DATE/, {hierarchy}/current -> RUN_DATE
# DAH has no hierarchy split (national-level data).
_A02_HIERARCHY   = _PROCESSED_STAGE / "hierarchy"   / LSAE_HIERARCHY
_A02_POPULATION  = _PROCESSED_STAGE / "population"  / LSAE_HIERARCHY
_A02_DAH         = _PROCESSED_STAGE / "covariates"  / "dah"
_A02_MAL_RAKED_AA = _PROCESSED_STAGE / "malaria"    / "raked_aa" / LSAE_HIERARCHY
_A02_MAL_RAKED_AS = _PROCESSED_STAGE / "malaria"    / "raked_as" / LSAE_HIERARCHY
_A02_DEN_RAKED_AA = _PROCESSED_STAGE / "dengue"     / "raked_aa" / LSAE_HIERARCHY
_A02_DEN_RAKED_AS = _PROCESSED_STAGE / "dengue"     / "raked_as" / LSAE_HIERARCHY

# Write paths (current run)
HIERARCHY_WRITE_PATH    = _artifact_write(_A02_HIERARCHY)
POPULATION_WRITE_PATH   = _artifact_write(_A02_POPULATION)
DAH_WRITE_PATH          = _artifact_write(_A02_DAH)
MAL_RAKED_AA_WRITE_PATH = _artifact_write(_A02_MAL_RAKED_AA)
MAL_RAKED_AS_WRITE_PATH = _artifact_write(_A02_MAL_RAKED_AS)
DEN_RAKED_AA_WRITE_PATH = _artifact_write(_A02_DEN_RAKED_AA)
DEN_RAKED_AS_WRITE_PATH = _artifact_write(_A02_DEN_RAKED_AS)

# Read paths (via current/ symlink)
HIERARCHY_READ_PATH    = _artifact_read(_A02_HIERARCHY)
POPULATION_READ_PATH   = _artifact_read(_A02_POPULATION)
DAH_READ_PATH          = _artifact_read(_A02_DAH)
MAL_RAKED_AA_READ_PATH = _artifact_read(_A02_MAL_RAKED_AA)
MAL_RAKED_AS_READ_PATH = _artifact_read(_A02_MAL_RAKED_AS)
DEN_RAKED_AA_READ_PATH = _artifact_read(_A02_DEN_RAKED_AA)
DEN_RAKED_AS_READ_PATH = _artifact_read(_A02_DEN_RAKED_AS)

# ── 03-modeling_data artifact roots ──────────────────────────────────────────
_A03_MAL_MODELING  = _MODELING_STAGE / "malaria"    / "modeling_dfs" / LSAE_HIERARCHY
_A03_DEN_MODELING  = _MODELING_STAGE / "dengue"     / "modeling_dfs" / LSAE_HIERARCHY

# Write paths
MAL_MODELING_WRITE_PATH  = _artifact_write(_A03_MAL_MODELING)
DEN_MODELING_WRITE_PATH  = _artifact_write(_A03_DEN_MODELING)

# Read paths
MAL_MODELING_READ_PATH   = _artifact_read(_A03_MAL_MODELING)
DEN_MODELING_READ_PATH   = _artifact_read(_A03_DEN_MODELING)

# ── Stage-level paths (stages 04–10, not yet artifact-structured) ─────────────
FORECASTING_DATA_PATH = _FORECASTING_STAGE / RUN_DATE
UPLOAD_DATA_PATH      = _UPLOAD_STAGE      / RUN_DATE
VISUALIZATION_PATH    = _VIZ_STAGE         / RUN_DATE
FIGURES_PATH          = _FIGURES_STAGE     / RUN_DATE
MANUSCRIPT_PATH       = _MANUSCRIPT_STAGE  / RUN_DATE
PRESENTATION_PATH     = _PRESENTATION_STAGE / RUN_DATE

FORECASTING_DATA_READ_PATH = _FORECASTING_STAGE / "current"

FHS_RESULTS_PATH = Path('/mnt/share/forecasting/data/9/future')
RR_PATH = Path('/mnt/team/rapidresponse/pub/malaria-denv')

repo_name = "idd-forecast-mbp"
package_name = "idd_forecast_mbp"


# Constants
# hierarchies = ["lsae_1209", "gbd_2021", "lsae_1285", "gbd_2023"]
hierarchies = [LSAE_HIERARCHY, "gbd_2023"]
hierarchies = [LSAE_HIERARCHY]
years = list(range(1970, 2101))
past_years = list(range(1970, 2024))
model_years = list(range(2000, 2101))
future_years = list(range(2024, 2101))

# GBD Constants
gbd_constants = {
        "release_2021_id": 9,
        "release_2023_id": 16,
        "como_2023_v": 1762,
        "codcorrect_2023_v": 528,
        "dalynator_2023_v": 102,
        "burdenator_2023_v": 395,
        "compare_2023_v": 8352  
}
# como_2023_v = 1591
# codcorrect_2023_v = 461
# dalynator_2023_v = 96
# burdenator_2023_v = 360
# compare_2023_v = 8234


ages = 22
sexes = 3
dengue_id = 357
malaria_id = 345
malaria_pf_id = 856
malaria_pv_id = 857


aa_merge_variables = ["location_id", "year_id"]
as_merge_variables = ["location_id", "year_id", "age_group_id", "sex_id"]
#
draws = [f"{i:03d}" for i in range(100)]
fhs_draws = [f"draw_{i}" for i in range(100)]

# Maps for various constants
cause_map = {
    'malaria':{
        'cause_id': malaria_id,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Malaria',
        'fhs_cause_name': 'malaria'
    },
    'malaria_pf':{
        'cause_id': malaria_pf_id,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Malaria falciparum',
        'fhs_cause_name': 'malaria_falciparum'
    },
    'malaria_pv':{
        'cause_id': malaria_pv_id,
        'reference_age_group_id': 3,
        'reference_sex_id': 1,
        'cause_name': 'Malaria vivax',
        'fhs_cause_name': 'malaria_vivax'
    },
    'dengue': {
        'cause_id': dengue_id,
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
        "name": "RCP2.6",
        "rcp_scenario": 2.6,
        "color": "#046C9A",
        "dhs_scenario": 66,
        "dhs_vbd_scenario": 75
    },
    "ssp245": {
        "name": "RCP4.5",
        "rcp_scenario": 4.5,
        "color": "#E58601",
        "dhs_scenario": 0,
        "dhs_vbd_scenario": 0
    },
    "ssp585": {
        "name": "RCP8.5",
        "rcp_scenario": 8.5,
        "color": "#A42820",
        "dhs_scenario": 54,
        "dhs_vbd_scenario": 76
    }
}

ssp_scenarios = {
    "ssp126": {
        "name": "RCP2.6",
        "rcp_scenario": 2.6,
        "color": "#046C9A",
        "dhs_scenario": 66
    },
    "ssp245": {
        "name": "RCP4.5",
        "rcp_scenario": 4.5,
        "color": "#E58601",
        "dhs_scenario": 0
    },
    "ssp585": {
        "name": "RCP8.5",
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


covariate_map = {
    'suitability' : {
        'var': 'suitability',
        'ylabel': 'Suitability (days)',
        'title' : 'Temperature Suitability'
    },
    'gdppc_mean': {
        'var': 'gdppc_mean',
        'ylabel': 'GDP per Capita',
        'title' : 'GDP per Capita (in 2020 USD)'
    },
    'dah_pc': {
        'var': 'dah_pc',
        'ylabel': 'DAH per Capita',
        'title' : 'DAH per Capita (in 2020 USD)'
    },
    'flooding_pc': {
        'var': 'flooding_pc',
        'ylabel': 'Flood days per capita',
        'title' : 'flood days per capita'
    },
    'urbanization': {
        'var': 'urbanization',
        'ylabel': 'Urbanization (%)',
        'title' : 'Urbanization'
    }
}