################################################################
### AGE-SPECIFIC POPULATION DATA PREPARATION
################################################################

###----------------------------------------------------------###
### 1. Setup and Initialization
### Establishes the required libraries, constants, and mapping dictionaries.
### Defines the file paths and data structures for population processing.
###----------------------------------------------------------###
import pandas as pd
import numpy as np
import itertools
import os
import sys
import xarray as xr
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.xarray_functions import read_netcdf_with_integer_ids, write_netcdf, convert_with_preset

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

RAW_DATA_PATH = rfc.MODEL_ROOT / "01-raw_data"
PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"

GBD_DATA_PATH = f"{RAW_DATA_PATH}/gbd"
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet'
age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"
age_sex_df_path = f'{PROCESSED_DATA_PATH}/age_sex_df.parquet'
fhs_hierarchy_df_path = f"{GBD_DATA_PATH}/fhs_2023_modeling_hierarchy.parquet"

gbd_population_path = f"{GBD_DATA_PATH}/gbd_2023_population.parquet"
aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population_df.parquet"
as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population_df.parquet"
aa_full_population_ds_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population_ds.nc"
as_full_population_ds_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population_ds.nc"

aa_fhs_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_fhs_population_df.parquet"
as_fhs_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_fhs_population_df.parquet"
aa_fhs_population_ds_path = f"{PROCESSED_DATA_PATH}/aa_2023_fhs_population_ds.nc"
as_fhs_population_ds_path = f"{PROCESSED_DATA_PATH}/as_2023_fhs_population_ds.nc"

missing_level_4_location_path = f"{PROCESSED_DATA_PATH}/missing_level_4_location_ids.parquet"
missing_level_5_location_path = f"{PROCESSED_DATA_PATH}/missing_level5_location_ids.parquet"
###----------------------------------------------------------###
### 2. Hierarchy Data Loading
### Loads and filters geographic hierarchies for both LSAE and FHS systems.
### These hierarchies define the spatial structure for population allocation.
###----------------------------------------------------------###
# Load hierarchy data
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)
# hierarchy_df = hierarchy_df[hierarchy_df["level"] >= 3]
#
fhs_hierarchy_df = read_parquet_with_integer_ids(fhs_hierarchy_df_path)
# fhs_hierarchy_df = fhs_hierarchy_df[fhs_hierarchy_df["level"] >= 3]

age_metadata_df = read_parquet_with_integer_ids(age_metadata_path)
age_group_ids = age_metadata_df["age_group_id"].unique().tolist()
age_group_filter = ('age_group_id', 'in', age_group_ids)
aa_age_group_filter = ('age_group_id', 'in', [22])  # All ages
sex_ids = [1, 2]
sex_filter = ('sex_id', 'in', sex_ids)
all_sex_filter = ('sex_id', 'in', [3])  # All sex

as_merge_variables = rfc.as_merge_variables

gbd_population_df = read_parquet_with_integer_ids(gbd_population_path)
# Only keep location_ids that are in the hierarchy_df
gbd_population_df = gbd_population_df[gbd_population_df["location_id"].isin(hierarchy_df["location_id"])]

##################
# Past FHS Population Data Loading
##################
past_fhs_population_path = f"{GBD_DATA_PATH}/fhs_2023_population.parquet"
as_past_fhs_population_df = read_parquet_with_integer_ids(past_fhs_population_path,
    filters=[[age_group_filter, sex_filter]])

aa_past_fhs_population_df = read_parquet_with_integer_ids(past_fhs_population_path,
    filters=[[aa_age_group_filter, all_sex_filter]])

##################
# Future FHS Population Data Loading
##################
future_fhs_population_path = "/mnt/share/forecasting/data/33/future/population/20250319_updated_rerun_pop_shifted_etl_417/population.nc"
# Select only location_id 4756 and age_group_id 235
as_future_fhs_population = xr.open_dataset(future_fhs_population_path).sel(
    age_group_id=age_metadata_df["age_group_id"].unique(),
    sex_id=sex_ids
).population.mean(dim='draw')
as_future_fhs_population_df = as_future_fhs_population.to_dataframe().reset_index()
# drop the scenario
as_future_fhs_population_df = as_future_fhs_population_df.drop(columns=["scenario"])

aa_future_fhs_population = xr.open_dataset(future_fhs_population_path).sel(
    age_group_id=22,
    sex_id=3
).population.mean(dim='draw')
aa_future_fhs_population_df = aa_future_fhs_population.to_dataframe().reset_index()
# drop the scenario
aa_future_fhs_population_df = aa_future_fhs_population_df.drop(columns=["scenario"])
##################
# Fix the 44858 to 60908; 95069; 94364 issue
##################
df_2023 = as_past_fhs_population_df[(as_past_fhs_population_df["year_id"] == 2023) & (as_past_fhs_population_df["location_id"].isin([60908, 95069, 94364]))].copy()
df_44858 = as_future_fhs_population_df[as_future_fhs_population_df["location_id"] == 44858].copy()
df_2023 = df_2023.drop(columns=["year_id"])
df_44858 = df_44858.drop(columns=["location_id"])
df_44858 = df_44858.rename(columns={"population": "population_44858"})
df_2023_sum = df_2023.groupby(["age_group_id", "sex_id"]).agg({"population": "sum"}).reset_index()
df_2023_sum = df_2023_sum.rename(columns={"population": "total_as_population"})
df_2023 = df_2023.merge(df_2023_sum, on=["age_group_id", "sex_id"], how="left")
df_2023["as_fraction"] = df_2023["population"] / df_2023["total_as_population"]
df_2023 = df_2023[["age_group_id", "sex_id", "location_id", "as_fraction"]]
df_as_new_locations = df_2023.merge(df_44858, on=["age_group_id","sex_id"], how="left")
df_as_new_locations["population"] = df_as_new_locations["as_fraction"] * df_as_new_locations["population_44858"]
df_as_new_locations = df_as_new_locations.drop(columns=["population_44858", "as_fraction"])
df_aa_new_locations = df_as_new_locations.groupby(["location_id", "year_id"]).agg({"population": "sum"}).reset_index().copy()
df_aa_new_locations["age_group_id"] = 22
df_aa_new_locations["sex_id"] = 3
# Drop 44858 from as_future_fhs_population_df and aa_future_fhs_population_df
as_future_fhs_population_df = as_future_fhs_population_df[as_future_fhs_population_df["location_id"] != 44858]
aa_future_fhs_population_df = aa_future_fhs_population_df[aa_future_fhs_population_df["location_id"] != 44858]
# Append the new locations to the future FHS population dataframes
as_future_fhs_population_df = pd.concat([as_future_fhs_population_df, df_as_new_locations], ignore_index=True)
aa_future_fhs_population_df = pd.concat([aa_future_fhs_population_df, df_aa_new_locations], ignore_index=True)


##################
# Combine Past and Future FHS Population Data
##################
# All-age FHS population data
aa_fhs_population_df = pd.concat([aa_past_fhs_population_df, aa_future_fhs_population_df], ignore_index=True)

# Age-specific FHS population data
as_fhs_population_df = pd.concat([as_past_fhs_population_df, as_future_fhs_population_df], ignore_index=True)

##################
# Combine AA and AS FHS Population Data
##################
aa_fhs_population_df = aa_fhs_population_df.rename(columns={
    "population": "aa_population"
})

as_fhs_population_df = as_fhs_population_df.merge(
    aa_fhs_population_df[["location_id", "year_id", "aa_population"]],
    on=["location_id", "year_id"],
    how="left"
)

as_fhs_population_df["as_population_fraction"] = as_fhs_population_df["population"] / as_fhs_population_df["aa_population"]


# Write to parquet
write_parquet(aa_fhs_population_df, aa_fhs_population_df_path)
write_parquet(as_fhs_population_df, as_fhs_population_df_path)

aa_fhs_population_ds = convert_with_preset(aa_fhs_population_df, preset='aa_variables')
write_netcdf(aa_fhs_population_ds, aa_fhs_population_ds_path)
as_fhs_population_ds = convert_with_preset(as_fhs_population_df, preset='as_variables')
write_netcdf(as_fhs_population_ds, as_fhs_population_ds_path)


###----------------------------------------------------------###
### 3. Base Population Data Loading
### Loads and filters LSAE population data to prepare for age-specific disaggregation.
### This establishes the total population values that will be distributed across age groups.
###----------------------------------------------------------###

# Create aa lsae population
lsae_years = list(range(2000, 2024))

lsae_population_dfs = []
for year in lsae_years:
    adm1_path = f"/mnt/share/geospatial/ihmepop/gbd_release_id_16/lsae_1209/pop_agg/{year}q1/aggregations/adm1.csv"
    adm2_path = f"/mnt/share/geospatial/ihmepop/gbd_release_id_16/lsae_1209/pop_agg/{year}q1/aggregations/adm2.csv"
    adm1_df = pd.read_csv(adm1_path)
    adm2_df = pd.read_csv(adm2_path)
    lsae_population_df = pd.concat([adm1_df, adm2_df], ignore_index=True)
    # Drop count, var, and location_name columns
    lsae_population_df = lsae_population_df.drop(columns=["count", "var", "location_name"], errors='ignore')
    # Rename columns to match hierarchy_df
    lsae_population_df = lsae_population_df.rename(columns={
        "Location ID": "location_id",
        "sum": "population"
    })
    lsae_population_df["year_id"] = year
    lsae_population_dfs.append(lsae_population_df)

# Concatenate all years into a single DataFrame
aa_full_population_df = pd.concat(lsae_population_dfs, ignore_index=True)




########################
## Look for missing locations in the lsae population data
########################

missing_level_4_location_ids = hierarchy_df[
    (hierarchy_df["level"] == 4) &
    (~hierarchy_df["location_id"].isin(aa_full_population_df["location_id"]))
]["location_id"].unique().tolist()

missing_level_5_location_ids = hierarchy_df[
    (hierarchy_df["level"] == 5) &
    (~hierarchy_df["location_id"].isin(aa_full_population_df["location_id"]))
]["location_id"].unique().tolist()

missing_location_ids = missing_level_4_location_ids + missing_level_5_location_ids
if not missing_location_ids:
    print("No missing locations found in LSAE population data.")
else:
    print(f"Missing locations found in LSAE population data: {len(missing_location_ids)}")

    if not missing_level_4_location_ids:
        print("No missing level 4 locations found in LSAE population data.")
    else:
        print(f"Missing level 4 locations found in LSAE population data: {len(missing_level_4_location_ids)}")
        missing_level_4_rows = hierarchy_df[hierarchy_df["location_id"].isin(missing_level_4_location_ids)]
        write_parquet(missing_level_4_rows, missing_level_4_location_path)

    if not missing_level_5_location_ids:
        print("No missing level 5 locations found in LSAE population data.")
    else:
        print(f"Missing level 5 locations found in LSAE population data: {len(missing_level_5_location_ids)}")
        missing_level_5_rows = hierarchy_df[hierarchy_df["location_id"].isin(missing_level_5_location_ids)]
        write_parquet(missing_level_5_rows, missing_level_5_location_path)

    # Update the hierarchy_df to indicate which locations are in GBD but not in LSAE
    hierarchy_df['in_gbd_not_lsae'] = hierarchy_df['location_id'].isin(missing_location_ids)
    ##########
    ## Get all age population from GBD population data
    ##########

    missing_population_df = gbd_population_df[
        (gbd_population_df["location_id"].isin(missing_location_ids)) &
        (gbd_population_df["year_id"].isin(lsae_years))
    ].copy()

    missing_aa_population_df = missing_population_df[
        (missing_population_df["age_group_id"] == 22) &  # All ages
        (missing_population_df["sex_id"] == 3)].drop(columns=["age_group_id", "sex_id"])
    # Append missing population to aa lsae population
    aa_full_population_df = pd.concat([aa_full_population_df, missing_aa_population_df], ignore_index=True)

    # Look for more missing locations in the hierarchy_df

    still_missing = hierarchy_df[(hierarchy_df["level"].isin([4,5])) &
        (~hierarchy_df["location_id"].isin(aa_full_population_df["location_id"]))]
    if still_missing.empty:
        print("No still missing locations found.")
    else:
        print(f"Found {len(still_missing)} still missing locations.")
        hierarchy_df['no_info'] = hierarchy_df['location_id'].isin(still_missing)
        # Create a DataFrame for still missing locations with population 0


        # Create all combinations of locations and years
        combinations = list(itertools.product(
            still_missing["location_id"].unique(),
            lsae_years
        ))

        still_missing_df = pd.DataFrame(combinations, columns=["location_id", "year_id"])
        still_missing_df["population"] = 0

        # Append still missing locations with population 0 to aa lsae population
        aa_full_population_df = pd.concat([aa_full_population_df, still_missing_df], ignore_index=True)
        # Write the aa lsae population to parquet

    write_parquet(hierarchy_df, hierarchy_df_path)


aa_gbd_population_df = gbd_population_df[(gbd_population_df['age_group_id'] == 22) & (gbd_population_df['sex_id'] == 3)].copy()
aa_gbd_population_df = aa_gbd_population_df.drop(columns=['age_group_id','sex_id'])
aa_gbd_population_df = aa_gbd_population_df.rename(columns={'population': 'aa_population'})

as_gbd_population_df = gbd_population_df[(gbd_population_df['age_group_id'].isin(age_group_ids)) & (gbd_population_df['sex_id'].isin(sex_ids))].copy()
as_gbd_population_df = as_gbd_population_df.merge(aa_gbd_population_df, on=['location_id', 'year_id'], how='left')
as_gbd_population_df['as_population_fraction'] = as_gbd_population_df['population'] / as_gbd_population_df['aa_population']

##################
# Rake LSAE to FHS
##################
# Step 0: Prep with GBD population data
aa_full_population_df = aa_full_population_df.merge(
    aa_gbd_population_df[['location_id', 'year_id', 'aa_population']],
    on=['location_id', 'year_id'],
    how='left'
)
# Replace population with aa_population where available, otherwise keep original
aa_full_population_df['population'] = aa_full_population_df['aa_population'].fillna(
    aa_full_population_df['population']
)
# Drop the temporary aa_population column
aa_full_population_df = aa_full_population_df.drop(columns=['aa_population'])

# Step 1: Merge and replace where they overlap
aa_full_population_df = aa_full_population_df.merge(
    aa_fhs_population_df[['location_id', 'year_id', 'aa_population']],
    on=['location_id', 'year_id'],
    how='left'
)
# Replace population with aa_population where available, otherwise keep original
aa_full_population_df['population'] = aa_full_population_df['aa_population'].fillna(
    aa_full_population_df['population']
)
# Drop the temporary aa_population column
aa_full_population_df = aa_full_population_df.drop(columns=['aa_population'])
aa_full_population_df = aa_full_population_df.merge(
    hierarchy_df[['location_id', 'level', 'parent_id']],
    on='location_id',
    how='left'
)

# Step 2: Aggregate level 4 to level 3 and rake
aa_lsae_level_4_population_df = aa_full_population_df[aa_full_population_df['level'] == 4].copy()
tmp_agg = aa_lsae_level_4_population_df.groupby(['parent_id', 'year_id']).agg({'population': 'sum'}).reset_index()
tmp_agg = tmp_agg.rename(columns={'parent_id': 'location_id'})
tmp_agg = tmp_agg.merge(aa_fhs_population_df[['location_id', 'year_id', 'aa_population']],
                        on=['location_id', 'year_id'],
                        how='left')
tmp_agg['raking_factor'] = tmp_agg['aa_population'] / tmp_agg['population']
tmp_agg.loc[tmp_agg['aa_population'] == 0, 'raking_factor'] = 1
tmp_agg = tmp_agg.rename(columns={'location_id': 'parent_id'})
aa_lsae_level_4_population_df = aa_lsae_level_4_population_df.merge(
    tmp_agg[['parent_id', 'year_id', 'raking_factor']],
    on=['parent_id', 'year_id'],
    how='left'
)

aa_lsae_level_4_population_df['population'] = aa_lsae_level_4_population_df['population'] * aa_lsae_level_4_population_df['raking_factor']
aa_lsae_level_4_population_df = aa_lsae_level_4_population_df.drop(columns=['raking_factor', 'level', 'parent_id'])

# Step 3: Aggregate level 5 to level 4 and rake
aa_lsae_level_5_population_df = aa_full_population_df[aa_full_population_df['level'] == 5].copy()
tmp_agg = aa_lsae_level_5_population_df.groupby(['parent_id', 'year_id']).agg({'population': 'sum'}).reset_index()
tmp_agg = tmp_agg.rename(columns={'parent_id': 'location_id'})
tmp_level_4 = aa_lsae_level_4_population_df.copy()
tmp_level_4 = tmp_level_4.rename(columns={'population': 'aa_population'})
tmp_agg = tmp_agg.merge(tmp_level_4[['location_id', 'year_id', 'aa_population']],
                        on=['location_id', 'year_id'],
                        how='left')
tmp_agg['raking_factor'] = tmp_agg['aa_population'] / tmp_agg['population']
# Set raking factor to 1 if aa_popultion is 0
tmp_agg.loc[tmp_agg['aa_population'] == 0, 'raking_factor'] = 1
tmp_agg = tmp_agg.rename(columns={'location_id': 'parent_id'})
aa_lsae_level_5_population_df = aa_lsae_level_5_population_df.merge(
    tmp_agg[['parent_id', 'year_id', 'raking_factor']],
    on=['parent_id', 'year_id'],
    how='left'
)

aa_lsae_level_5_population_df['population'] = aa_lsae_level_5_population_df['population'] * aa_lsae_level_5_population_df['raking_factor']
aa_lsae_level_5_population_df = aa_lsae_level_5_population_df.drop(columns=['raking_factor', 'level', 'parent_id'])

# Step 4: Combine level 0:3, 4, and 5 populations
aa_fhs_level_0_3_population_df = aa_fhs_population_df.copy()
aa_fhs_level_0_3_population_df = aa_fhs_level_0_3_population_df.merge(
    hierarchy_df[['location_id', 'level']],
    on='location_id',
    how='left'
)
aa_fhs_level_0_3_population_df = aa_fhs_level_0_3_population_df[aa_fhs_level_0_3_population_df['level'] <= 3].copy()
aa_fhs_level_0_3_population_df = aa_fhs_level_0_3_population_df.drop(columns=['age_group_id', 'sex_id', 'level'])
aa_fhs_level_0_3_population_df = aa_fhs_level_0_3_population_df.rename(columns={'aa_population': 'population'})

level_4_last_df = aa_lsae_level_4_population_df[aa_lsae_level_4_population_df['year_id'] == 2023].copy()
level_5_last_df = aa_lsae_level_5_population_df[aa_lsae_level_5_population_df['year_id'] == 2023].copy()
subnat_last_df = pd.concat([level_4_last_df, level_5_last_df], ignore_index=True)

subnat_last_df = subnat_last_df.merge(
    hierarchy_df[['location_id', 'fhs_location_id']],
    on='location_id',
    how='left')

aa_fhs_df = aa_fhs_population_df.copy()
aa_fhs_df = aa_fhs_df.rename(
    columns={'location_id': 'fhs_location_id',
             'aa_population': 'fhs_population'})

aa_fhs_df = aa_fhs_df[aa_fhs_df['fhs_location_id'].isin(subnat_last_df['fhs_location_id'].unique().tolist())].copy()
aa_fhs_df = aa_fhs_df.drop(columns=['age_group_id', 'sex_id'])

subnat_last_df = subnat_last_df.merge(
    aa_fhs_df,
    on=['fhs_location_id', 'year_id'],
    how='left'
)
subnat_last_df['fhs_fraction'] = subnat_last_df['population'] / subnat_last_df['fhs_population']
subnat_last_df = subnat_last_df[['location_id', 'fhs_fraction']]

lsae_level_4_location_ids = hierarchy_df[hierarchy_df['level'] == 4]['location_id'].unique().tolist()
lsae_level_5_location_ids = hierarchy_df[hierarchy_df['level'] == 5]['location_id'].unique().tolist()
future_year_ids = list(range(2024, 2101))

combinations = list(itertools.product(lsae_level_4_location_ids + lsae_level_5_location_ids, future_year_ids))

subnat_future_population_df = pd.DataFrame(combinations, columns=["location_id", "year_id"])
subnat_future_population_df = subnat_future_population_df.merge(
    hierarchy_df[['location_id', 'fhs_location_id']],
    on='location_id',
    how='left')

subnat_future_population_df = subnat_future_population_df.merge(
    aa_fhs_df,
    on=['fhs_location_id', 'year_id'],
    how='left'
)
subnat_future_population_df = subnat_future_population_df.merge(
    subnat_last_df,
    on=['location_id'],
    how='left'
)

subnat_future_population_df['population'] = subnat_future_population_df['fhs_population'] * subnat_future_population_df['fhs_fraction']
subnat_future_population_df = subnat_future_population_df.drop(columns=['fhs_population', 'fhs_fraction', 'fhs_location_id'])

aa_full_population_df = pd.concat([
    aa_fhs_level_0_3_population_df,
    aa_lsae_level_4_population_df,
    aa_lsae_level_5_population_df,
    subnat_future_population_df
], ignore_index=True)

# Step 5: Finalize the aa_full_population_df
write_parquet(aa_full_population_df, aa_full_population_df_path)
# Convert to xarray dataset and write to netCDF
aa_full_population_ds = convert_with_preset(aa_full_population_df, preset='aa_variables')
write_netcdf(aa_full_population_ds, aa_full_population_ds_path)

###----------------------------------------------------------###
### 5. Age Metadata Processing
### Loads age group metadata and creates all possible age-sex combinations
### for disaggregation of all-age population values.
###----------------------------------------------------------###
combinations = list(itertools.product(age_group_ids, sex_ids))
age_sex_df = pd.DataFrame(combinations, columns=['age_group_id', 'sex_id'])
write_parquet(age_sex_df, age_sex_df_path)

###----------------------------------------------------------###
### 6. Age-Specific Population Generation for LSAE
### Creates age-specific population estimates for LSAE locations by applying
### demographic patterns from FHS locations to total population counts.
###----------------------------------------------------------###
# Subset aa_full_population_df to only include location_ids that aren't in aa_fhs_population_df
sub_aa_full_population_df = aa_full_population_df.copy()
sub_aa_full_population_df = sub_aa_full_population_df.merge(
    hierarchy_df[['location_id', 'level']],
    on='location_id',
    how='left')
sub_aa_full_population_df = sub_aa_full_population_df[sub_aa_full_population_df['level'] >= 3].copy()


sub_as_full_population_df = sub_aa_full_population_df.merge(age_sex_df, how = "cross")
sub_as_full_population_df = sub_as_full_population_df.rename(columns={"population": "aa_population"})
sub_as_full_population_df = sub_as_full_population_df.merge(hierarchy_df[["location_id", 'gbd_location_id', "fhs_location_id"]], on="location_id", how="left")

past_as_df = sub_as_full_population_df[sub_as_full_population_df["year_id"] <= 2023].copy()

as_gbd_population_df['set_by_gbd'] = True
past_as_df = past_as_df.merge(
    as_gbd_population_df[as_merge_variables + ['population', 'set_by_gbd']],
    on=as_merge_variables,
    how='left'
    )
    
# Track which locations are already set by GBD
mask = past_as_df['set_by_gbd'].isna()
past_as_df.loc[mask, 'set_by_gbd'] = False
past_as_df['set_by_gbd'] = past_as_df['set_by_gbd'].astype('boolean')

as_gbd_population_df = as_gbd_population_df.rename(columns={'location_id': 'gbd_location_id'})

past_as_df = past_as_df.merge(
    as_gbd_population_df[['gbd_location_id', 'year_id', 'age_group_id', 'sex_id', 'as_population_fraction']],
    on=['gbd_location_id', 'year_id', 'age_group_id', 'sex_id'],
    how='left'
    )

mask = past_as_df['set_by_gbd'] == False
past_as_df.loc[mask, 'population'] = (
    past_as_df.loc[mask, 'aa_population'] * 
    past_as_df.loc[mask, 'as_population_fraction']
)

past_as_df = past_as_df.drop(columns=['gbd_location_id', 'fhs_location_id', 'set_by_gbd', 'level'])

future_as_df = sub_as_full_population_df[sub_as_full_population_df["year_id"] >= 2024].copy()

as_fhs_population_df['set_by_fhs'] = True
future_as_df = future_as_df.merge(
    as_fhs_population_df[as_merge_variables + ['population', 'set_by_fhs']],
    on=as_merge_variables,
    how='left'
    )
    
# Track which locations are already set by GBD
mask = future_as_df['set_by_fhs'].isna()
future_as_df.loc[mask, 'set_by_fhs'] = False
future_as_df['set_by_fhs'] = future_as_df['set_by_fhs'].astype('boolean')

as_fhs_population_df = as_fhs_population_df.rename(columns={'location_id': 'fhs_location_id'})

future_as_df = future_as_df.merge(
    as_fhs_population_df[['fhs_location_id', 'year_id', 'age_group_id', 'sex_id', 'as_population_fraction']],
    on=['fhs_location_id', 'year_id', 'age_group_id', 'sex_id'],
    how='left'
    )

mask = future_as_df['set_by_fhs'] == False
future_as_df.loc[mask, 'population'] = (
    future_as_df.loc[mask, 'aa_population'] * 
    future_as_df.loc[mask, 'as_population_fraction']
)

future_as_df = future_as_df.drop(columns=['gbd_location_id', 'fhs_location_id', 'set_by_fhs', 'level'])

low_level_past = as_gbd_population_df.rename(columns={'gbd_location_id': 'location_id'}).copy()
low_level_past = low_level_past.merge(
    hierarchy_df[['location_id', 'level']],
    on='location_id',
    how='left'
)
low_level_past = low_level_past[low_level_past['level'] < 3].copy()
low_level_past = low_level_past.drop(columns=['level', 'set_by_gbd'])

low_level_future = as_fhs_population_df[as_fhs_population_df['year_id'] > 2023].rename(columns={'fhs_location_id': 'location_id'}).copy()
low_level_future = low_level_future.merge(
    hierarchy_df[['location_id', 'level']],
    on='location_id',
    how='left'
)
low_level_future = low_level_future[low_level_future['level'] < 3].copy()
low_level_future = low_level_future.drop(columns=['level', 'set_by_fhs'])

as_full_population_df = pd.concat([
    low_level_past,
    past_as_df,
    low_level_future,
    future_as_df],
    ignore_index=True
)
# As a last step, replace all age-sex population 

# Write the final DataFrame to a parquet file
write_parquet(as_full_population_df, as_full_population_df_path)
# Convert to xarray dataset and write to netCDF
as_full_population_ds = convert_with_preset(as_full_population_df, preset='as_variables')
write_netcdf(as_full_population_ds, as_full_population_ds_path)