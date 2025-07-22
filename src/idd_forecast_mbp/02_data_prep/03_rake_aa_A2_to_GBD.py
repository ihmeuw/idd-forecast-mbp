import numpy as np

import pandas as pd
from pathlib import Path
from datetime import datetime
from rra_tools.shell_tools import mkdir  # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import check_column_for_problematic_values
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.cause_processing_functions import format_aa_gbd_df, process_lsae_df
from idd_forecast_mbp.rake_and_aggregate_functions import rake_aa_count_lsae_to_gbd, make_aa_full_rate_df_from_aa_count_df, check_concordance, aggregate_aa_rate_lsae_to_gbd
from idd_forecast_mbp.xarray_functions import convert_to_xarray, write_netcdf

PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH

GBD_DATA_PATH = rfc.GBD_DATA_PATH
LSAE_INPUT_PATH = rfc.LSAE_INPUT_PATH

aa_full_malaria_df_path = PROCESSED_DATA_PATH / "aa_full_malaria_df.parquet"
aa_full_dengue_df_path = PROCESSED_DATA_PATH / "aa_full_dengue_df.parquet"
aa_full_malaria_ds_path = PROCESSED_DATA_PATH / "aa_full_malaria_ds.nc"
aa_full_dengue_ds_path = PROCESSED_DATA_PATH / "aa_full_dengue_ds.nc"
################################################################
#### Hierarchy Paths, loading, and cleaning
################################################################

aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population.parquet"
full_2023_hierarchy_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet"

hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)
aa_full_population_df = pd.read_parquet(aa_full_population_df_path)

aa_gbd_malaria_df_path = f"{GBD_DATA_PATH}/gbd_2023_malaria_aa.csv"
aa_gbd_dengue_df_path = f"{GBD_DATA_PATH}/gbd_2023_dengue_aa.csv"

measure_map = rfc.measure_map
ploblematic_rule_map = rfc.problematic_rule_map

################################################################
###  MALARIA DATA PROCESSING AND RAKING
################################################################

###----------------------------------------------------------###
### 1. Data Loading - Get raw data sources
### Imports raw malaria prevalence data (PfPR) and reference GBD datasets needed for raking.
### These foundational datasets provide the base inputs for all subsequent processing.
###----------------------------------------------------------###

# Load PfPR (parasite prevalence) and GBD reference data
aa_lsae_malaria_pfpr_df = process_lsae_df("malaria", "pfpr", aa_full_population_df, hierarchy_df)
aa_full_malaria_pfpr_df = aggregate_aa_rate_lsae_to_gbd(rate_variable = "malaria_pfpr", hierarchy_df = hierarchy_df, aa_lsae_rate_df = aa_lsae_malaria_pfpr_df, aa_full_population_df=aa_full_population_df, return_full_df = True)

# Load GBD reference data for raking
aa_gbd_malaria_df = pd.read_csv(aa_gbd_malaria_df_path, low_memory=False)

###----------------------------------------------------------###
### 2. Incidence Processing & Raking
### This section processes malaria incidence data, including counts and rates.
### It formats the GBD data, processes the LSAE data, and then rakes the incidence counts
### to the GBD data. Finally, it calculates incidence rates from the counts.
###----------------------------------------------------------###
cause = 'malaria'
measure = 'incidence'
short_measure = measure_map[measure]['short']
metric = 'count'
count_variable = f'{cause}_{short_measure}_{metric}'
rate_variable = f'{cause}_{short_measure}_rate'

problematic_rules = ploblematic_rule_map[cause][measure]

aa_gbd_count_df = format_aa_gbd_df(cause, measure, metric, aa_gbd_malaria_df)
aa_gbd_count_df = aa_gbd_count_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")
aa_lsae_count_df = process_lsae_df(cause, measure, aa_full_population_df, hierarchy_df)

aa_full_malaria_inc_count_df = rake_aa_count_lsae_to_gbd(count_variable = count_variable, 
                                                 hierarchy_df =hierarchy_df, 
                                                 aa_gbd_count_df = aa_gbd_count_df, 
                                                 aa_lsae_count_df = aa_lsae_count_df,
                                                 problematic_rules = problematic_rules,
                                                 aa_full_count_df_path = None, return_full_df=True)

check_column_for_problematic_values(count_variable, aa_full_malaria_inc_count_df)

aa_full_malaria_inc_rate_df = make_aa_full_rate_df_from_aa_count_df(rate_variable = rate_variable, 
                                                            count_variable = count_variable, 
                                                            aa_full_count_df = aa_full_malaria_inc_count_df, 
                                                            aa_full_population_df = aa_full_population_df, 
                                                            aa_full_rate_df_path=None, 
                                                            return_full_df=True)

check_column_for_problematic_values(rate_variable, aa_full_malaria_inc_rate_df)

aa_gbd_rate_df = format_aa_gbd_df(cause, measure, 'rate', aa_gbd_malaria_df)
aa_gbd_rate_df = aa_gbd_rate_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")

concordance_results = check_concordance(variable = rate_variable,
                                        aa_full_df = aa_full_malaria_inc_rate_df, 
                                        aa_gbd_df = aa_gbd_rate_df)
concordance_results

concordance_results = check_concordance(variable = count_variable,
                                        aa_full_df = aa_full_malaria_inc_count_df, 
                                        aa_gbd_df = aa_gbd_count_df)
concordance_results

###----------------------------------------------------------###
### 3. Mortality Processing & Raking
### Similar to incidence raking, this adjusts mortality estimates to match GBD death counts.
### Ensures mortality estimates are consistent with global health metrics while maintaining
### subnational distribution patterns.
###----------------------------------------------------------###
cause = 'malaria'
measure = 'mortality'
short_measure = measure_map[measure]['short']
metric = 'count'
count_variable = f'{cause}_{short_measure}_{metric}'
rate_variable = f'{cause}_{short_measure}_rate'

problematic_rules = ploblematic_rule_map[cause][measure]

aa_gbd_count_df = format_aa_gbd_df(cause, measure, metric, aa_gbd_malaria_df)
aa_gbd_count_df = aa_gbd_count_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")
aa_lsae_count_df = process_lsae_df(cause, measure, aa_full_population_df, hierarchy_df)

aa_full_malaria_mort_count_df = rake_aa_count_lsae_to_gbd(count_variable = count_variable, 
                                                 hierarchy_df =hierarchy_df, 
                                                 aa_gbd_count_df = aa_gbd_count_df, 
                                                 aa_lsae_count_df = aa_lsae_count_df,
                                                 problematic_rules = problematic_rules,
                                                 aa_full_count_df_path = None, return_full_df=True)

check_column_for_problematic_values(count_variable, aa_full_malaria_mort_count_df)

aa_full_malaria_mort_rate_df = make_aa_full_rate_df_from_aa_count_df(rate_variable = rate_variable, 
                                                            count_variable = count_variable, 
                                                            aa_full_count_df = aa_full_malaria_mort_count_df, 
                                                            aa_full_population_df = aa_full_population_df, 
                                                            aa_full_rate_df_path=None, 
                                                            return_full_df=True)

check_column_for_problematic_values(rate_variable, aa_full_malaria_mort_rate_df)

aa_gbd_rate_df = format_aa_gbd_df(cause, measure, 'rate', aa_gbd_malaria_df)
aa_gbd_rate_df = aa_gbd_rate_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")

concordance_results = check_concordance(variable = rate_variable,
                                        aa_full_df = aa_full_malaria_mort_rate_df, 
                                        aa_gbd_df = aa_gbd_rate_df)
concordance_results

concordance_results = check_concordance(variable = count_variable,
                                        aa_full_df = aa_full_malaria_mort_count_df, 
                                        aa_gbd_df = aa_gbd_count_df)
concordance_results

###----------------------------------------------------------###
### 4. Data Integration
### Combines all malaria metrics (prevalence, raked incidence, raked mortality) into a unified dataset. 
### This creates a comprehensive dataset with consistent metrics across all dimensions.
###----------------------------------------------------------###
files_to_merge = [
    aa_full_malaria_pfpr_df,
    aa_full_malaria_inc_rate_df,
    aa_full_malaria_inc_count_df,
    aa_full_malaria_mort_rate_df,
    aa_full_malaria_mort_count_df
]

# Merge all malaria dataframes together by location_id and year_id
potential_columns_to_drop = ['population', 'set_by_gbd']
aa_full_malaria_df = files_to_merge[0].copy()
for df in files_to_merge[1:]:
    # Check if either column is already in aa_full_malaria_df and drop it if it is
    cols_to_drop = [col for col in potential_columns_to_drop if col in df.columns and col in aa_full_malaria_df.columns]
    df = df.drop(columns=cols_to_drop)
    aa_full_malaria_df = pd.merge(aa_full_malaria_df, df, on=["location_id", "year_id"], how="left")

###----------------------------------------------------------###
### 5. Save Final Dataset
### Exports the final processed dataset to a parquet file for use in downstream modeling.
### This preserves the complete, harmonized dataset for forecasting applications.
###----------------------------------------------------------###
write_parquet(aa_full_malaria_df, aa_full_malaria_df_path)

################################################################
###  DENGUE DATA PROCESSING AND RAKING
################################################################

###----------------------------------------------------------###
### 1. Data Loading
### Imports dengue suitability data, which represents environmental suitability for 
### dengue transmission based on climate and other factors.
###----------------------------------------------------------###
# Load the dengue suitability data
aa_lsae_dengue_suit_df = process_lsae_df("dengue", "dengue_suitability", aa_full_population_df, hierarchy_df)
# Load GBD reference data for raking
aa_gbd_dengue_df = pd.read_csv(aa_gbd_dengue_df_path, low_memory=False)

###----------------------------------------------------------###
### 2. Incidence Processing & Raking
### This section processes dengue incidence data, including counts and rates.
### It formats the GBD data and then rakes the incidence counts
### to the GBD data. Finally, it calculates incidence rates from the counts.
###----------------------------------------------------------###
cause = 'dengue'
measure = 'incidence'
short_measure = measure_map[measure]['short']
metric = 'count'
count_variable = f'{cause}_{short_measure}_{metric}'
rate_variable = f'{cause}_{short_measure}_rate'

problematic_rules = ploblematic_rule_map[cause][measure]

aa_gbd_count_df = format_aa_gbd_df(cause, measure, metric, aa_gbd_dengue_df)
aa_gbd_count_df = aa_gbd_count_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")
aa_lsae_count_df = aa_lsae_dengue_suit_df.rename(columns={"dengue_suitability": "dengue_inc_count"}).copy()



aa_full_dengue_inc_count_df = rake_aa_count_lsae_to_gbd(count_variable = count_variable, 
                                                 hierarchy_df =hierarchy_df, 
                                                 aa_gbd_count_df = aa_gbd_count_df, 
                                                 aa_lsae_count_df = aa_lsae_count_df,
                                                 problematic_rules = problematic_rules,
                                                 aa_full_count_df_path = None, return_full_df=True)

check_column_for_problematic_values(count_variable, aa_full_dengue_inc_count_df)

aa_full_dengue_inc_rate_df = make_aa_full_rate_df_from_aa_count_df(rate_variable = rate_variable, 
                                                            count_variable = count_variable, 
                                                            aa_full_count_df = aa_full_dengue_inc_count_df, 
                                                            aa_full_population_df = aa_full_population_df, 
                                                            aa_full_rate_df_path=None, 
                                                            return_full_df=True)

check_column_for_problematic_values(rate_variable, aa_full_dengue_inc_rate_df)

aa_gbd_rate_df = format_aa_gbd_df(cause, measure, 'rate', aa_gbd_dengue_df)
aa_gbd_rate_df = aa_gbd_rate_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")

concordance_results = check_concordance(variable = rate_variable,
                                        aa_full_df = aa_full_dengue_inc_rate_df, 
                                        aa_gbd_df = aa_gbd_rate_df)
concordance_results

concordance_results = check_concordance(variable = count_variable,
                                        aa_full_df = aa_full_dengue_inc_count_df, 
                                        aa_gbd_df = aa_gbd_count_df)
concordance_results

###----------------------------------------------------------###
### 3. Mortality Processing & Raking
### Similar to incidence raking, this adjusts mortality estimates to match GBD death counts.
### Ensures mortality estimates are consistent with global health metrics while maintaining
### subnational distribution patterns.
###----------------------------------------------------------###


cause = 'dengue'
measure = 'mortality'
short_measure = measure_map[measure]['short']
metric = 'count'
count_variable = f'{cause}_{short_measure}_{metric}'
rate_variable = f'{cause}_{short_measure}_rate'

problematic_rules = ploblematic_rule_map[cause][measure]

aa_gbd_count_df = format_aa_gbd_df(cause, measure, metric, aa_gbd_dengue_df)
aa_gbd_count_df = aa_gbd_count_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")
aa_lsae_count_df = aa_lsae_dengue_suit_df.rename(columns={"dengue_suitability": "dengue_mort_count"}).copy()

aa_full_dengue_mort_count_df = rake_aa_count_lsae_to_gbd(count_variable = count_variable, 
                                                 hierarchy_df =hierarchy_df, 
                                                 aa_gbd_count_df = aa_gbd_count_df, 
                                                 aa_lsae_count_df = aa_lsae_count_df,
                                                 problematic_rules = problematic_rules,
                                                 aa_full_count_df_path = None, return_full_df=True)

check_column_for_problematic_values(count_variable, aa_full_dengue_mort_count_df)

aa_full_dengue_mort_rate_df = make_aa_full_rate_df_from_aa_count_df(rate_variable = rate_variable, 
                                                            count_variable = count_variable, 
                                                            aa_full_count_df = aa_full_dengue_mort_count_df, 
                                                            aa_full_population_df = aa_full_population_df, 
                                                            aa_full_rate_df_path=None, 
                                                            return_full_df=True)

check_column_for_problematic_values(rate_variable, aa_full_dengue_mort_rate_df)

aa_gbd_rate_df = format_aa_gbd_df(cause, measure, 'rate', aa_gbd_dengue_df)
aa_gbd_rate_df = aa_gbd_rate_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")

concordance_results = check_concordance(variable = rate_variable,
                                        aa_full_df = aa_full_dengue_mort_rate_df, 
                                        aa_gbd_df = aa_gbd_rate_df)
concordance_results

concordance_results = check_concordance(variable = count_variable,
                                        aa_full_df = aa_full_dengue_mort_count_df, 
                                        aa_gbd_df = aa_gbd_count_df)
concordance_results




aa_gbd_dengue_inc_count_df = format_aa_gbd_df("dengue", "incidence", "count", aa_gbd_dengue_df)
aa_gbd_dengue_inc_count_df = aa_gbd_dengue_inc_count_df.merge(
    aa_full_population_df,
    on=["location_id", "year_id"],
    how="left")

aa_lsae_dengue_inc_count_df = aa_lsae_dengue_suit_df.rename(columns={"dengue_suitability": "dengue_inc_count"}).copy()



aa_full_dengue_inc_count_df = rake_aa_count_lsae_to_gbd(count_variable = "dengue_inc_count", 
                                                 hierarchy_df =hierarchy_df, 
                                                 aa_gbd_count_df = aa_gbd_dengue_inc_count_df, 
                                                 aa_lsae_count_df = aa_lsae_dengue_inc_count_df,
                                                 problematic_rules = problematic_rules,
                                                 aa_full_count_df_path = None, return_full_df=True)

check_column_for_problematic_values('dengue_inc_count', aa_full_dengue_inc_count_df)

aa_full_dengue_inc_rate_df = make_aa_full_rate_df_from_aa_count_df(rate_variable = "dengue_inc_rate", 
                                                            count_variable = "dengue_inc_count", 
                                                            aa_full_count_df = aa_full_dengue_inc_count_df, 
                                                            aa_full_population_df = aa_full_population_df, 
                                                            aa_full_rate_df_path=None, 
                                                            return_full_df=True)

check_column_for_problematic_values('dengue_inc_rate', aa_full_dengue_inc_rate_df)

aa_gbd_dengue_inc_rate_df = format_aa_gbd_df('dengue', 'incidence', 'rate', aa_gbd_dengue_df)

concordance_results = check_concordance(variable = "dengue_inc_rate",
                                        aa_full_df = aa_full_dengue_inc_rate_df, 
                                        aa_gbd_df = aa_gbd_dengue_inc_rate_df)
concordance_results

concordance_results = check_concordance(variable = "dengue_inc_count",
                                        aa_full_df = aa_full_dengue_inc_count_df, 
                                        aa_gbd_df = aa_gbd_dengue_inc_count_df)
concordance_results


###----------------------------------------------------------###
### 4. Data Integration
### Combines all dengue metrics (raked incidence, raked mortality) into a unified dataset. 
### This creates a comprehensive dataset with consistent metrics across all dimensions.
###----------------------------------------------------------###
files_to_merge = [
    aa_full_dengue_inc_count_df,
    aa_full_dengue_inc_rate_df,
    aa_full_dengue_mort_count_df,
    aa_full_dengue_mort_rate_df
]

# Merge all dengue dataframes together by location_id and year_id
potential_columns_to_drop = ['population', 'set_by_gbd']
aa_full_dengue_df = files_to_merge[0].copy()
for df in files_to_merge[1:]:
    # Check if either column is already in aa_full_dengue_df and drop it if it is
    cols_to_drop = [col for col in potential_columns_to_drop if col in df.columns and col in aa_full_dengue_df.columns]
    df = df.drop(columns=cols_to_drop)
    aa_full_dengue_df = pd.merge(aa_full_dengue_df, df, on=["location_id", "year_id"], how="left")

###----------------------------------------------------------###
### 5. Save Final Dataset
### Exports the final processed dataset to a parquet file for use in downstream modeling.
### This preserves the complete, harmonized dataset for forecasting applications.
###----------------------------------------------------------###
write_parquet(aa_full_dengue_df, aa_full_dengue_df_path)

###----------------------------------------------------------###
### 6. Convert to xarray Dataset and save to NetCDF
### Converts the final dengue and malaria datasets to xarray Datasets
### and saves them to NetCDF files for efficient storage and access.
###----------------------------------------------------------###


aa_full_dengue_df = aa_full_dengue_df.set_index(['location_id', 'year_id']).sort_index()

# Use index.get_level_values instead of column access
all_locations = aa_full_dengue_df.index.get_level_values('location_id').unique()
all_years = aa_full_dengue_df.index.get_level_values('year_id').unique()
full_grid = pd.MultiIndex.from_product([all_locations, all_years], names=['location_id', 'year_id'])

# Get actual pairs
actual_pairs = aa_full_dengue_df.index

# Find missing pairs
missing_pairs = full_grid.difference(actual_pairs)
missing_df = pd.DataFrame(missing_pairs.to_list(), columns=['location_id', 'year_id'])

filled_rows = []
for _, row in missing_df.iterrows():
    loc = row['location_id']
    year = row['year_id']
    prev_year = year - 1
    try:
        prev_row = aa_full_dengue_df.loc[(loc, prev_year)]
        filled = prev_row.copy()
        filled.name = (loc, year)
        filled_rows.append(filled)
    except KeyError:
        zero_row = {col: 0 for col in aa_full_dengue_df.columns}
        filled_rows.append(pd.Series(zero_row, name=(loc, year)))

if filled_rows:
    filled_df = pd.DataFrame(filled_rows)
    filled_df.index = pd.MultiIndex.from_tuples(filled_df.index, names=['location_id', 'year_id'])
    aa_full_dengue_df = pd.concat([aa_full_dengue_df, filled_df])
    aa_full_dengue_df = aa_full_dengue_df.reset_index()


malaria_ds = convert_to_xarray(
    aa_full_malaria_df,
    dimensions=['location_id', 'year_id'],
    dimension_dtypes={'location_id': 'int32', 'year_id': 'int16'},
    auto_optimize_dtypes=True
)
write_netcdf(
    malaria_ds,
    aa_full_malaria_ds_path,
    compression=True,
    compression_level=4,
    chunking=True,
    chunk_by_dim={'location_id': 1500, 'year_id': 79},
    engine='netcdf4'
)

dengue_ds = convert_to_xarray(
    aa_full_dengue_df,
    dimensions=['location_id', 'year_id'],
    dimension_dtypes={'location_id': 'int32', 'year_id': 'int16'},
    auto_optimize_dtypes=True
)
write_netcdf(
    dengue_ds,
    aa_full_dengue_ds_path,
    compression=True,
    compression_level=4,
    chunking=True,
    chunk_by_dim={'location_id': 1500, 'year_id': 79},
    engine='netcdf4'
)