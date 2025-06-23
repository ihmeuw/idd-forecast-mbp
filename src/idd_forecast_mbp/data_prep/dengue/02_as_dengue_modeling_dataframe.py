import pandas as pd
import numpy as np
import os
import sys
import itertools
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import read_parquet_with_integer_ids

PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"
MODELING_DATA_PATH = rfc.MODEL_ROOT / "03-modeling_data"
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"

cause = "dengue"
cause_id = rfc.cause_map[cause]["cause_id"]

aa_fhs_data_path_template = "{FHS_DATA_PATH}/aa_cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_fhs.parquet"
as_fhs_data_path_template = "{FHS_DATA_PATH}/as_cause_id_{cause_id}_measure_id_{measure_id}_metric_id_{metric_id}_fhs.parquet"

as_lsae_population_path = f"{MODELING_DATA_PATH}/as_lsae_population_df.parquet"

age_metadata_path = f"{FHS_DATA_PATH}/age_metadata.parquet"

dengue_stage_2_df_path = f"{MODELING_DATA_PATH}/dengue_stage_2_modeling_df.parquet"
df = read_parquet_with_integer_ids(dengue_stage_2_df_path)

year_ids = df.year_id.unique()
lsae_location_ids = df.location_id.unique()
fhs_location_ids = df.fhs_location_id.unique()

year_filter = ('year_id', 'in', year_ids)
lsae_location_filter = ('location_id', 'in', lsae_location_ids)
fhs_location_filter = ('location_id', 'in', fhs_location_ids)

as_lsae_population = pd.read_parquet(as_lsae_population_path,
                                     filters=[[year_filter, lsae_location_filter]])

hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

age_metadata_df = read_parquet_with_integer_ids(age_metadata_path)
age_group_ids = age_metadata_df["age_group_id"].unique()
sex_ids = [1, 2]  # 1
combinations = list(itertools.product(age_group_ids, sex_ids))
as_df = pd.DataFrame(combinations, columns=['age_group_id', 'sex_id'])

df.rename(
    columns={
        "population": "aa_population",
    },
    inplace=True,
)

df = df.merge(
    as_df,
    how = "cross")

df = df.merge(
    as_lsae_population[["location_id", "year_id", "age_group_id", "sex_id", "population"]],
    how = "left",
    on = ["location_id", "year_id", "age_group_id", "sex_id"])

# Get dengue mortality counts for age specific
as_fhs_df_path = as_fhs_data_path_template.format(
    FHS_DATA_PATH=FHS_DATA_PATH,
    cause_id=cause_id,
    measure_id=1,
    metric_id=1
)
as_fhs_df = read_parquet_with_integer_ids(as_fhs_df_path,
                                     filters=[[year_filter, fhs_location_filter]])
# Rename location_id to fhs_location_id

as_fhs_df = as_fhs_df.rename(columns={
    "location_id": "fhs_location_id",
    "val": "as_fhs_dengue_mort_count"})

df = df.merge(
    as_fhs_df[["fhs_location_id", "year_id",
               "age_group_id","sex_id",
               "as_fhs_dengue_mort_count"]],
    how = "left",
    on = ["fhs_location_id", "year_id", 
          "age_group_id", "sex_id"])


# Get dengue mortality counts for age specific
aa_fhs_df_path = aa_fhs_data_path_template.format(
    FHS_DATA_PATH=FHS_DATA_PATH,
    cause_id=cause_id,
    measure_id=1,
    metric_id=1
)
aa_fhs_df = read_parquet_with_integer_ids(aa_fhs_df_path,
                                     filters=[[year_filter, fhs_location_filter]])
# Rename location_id to fhs_location_id
aa_fhs_df = aa_fhs_df.rename(columns={
    "location_id": "fhs_location_id",
    "val": "aa_fhs_dengue_mort_count"})

df = df.merge(
    aa_fhs_df[["fhs_location_id", "year_id",
                       "aa_fhs_dengue_mort_count"]],
    how = "left",
    on = ["fhs_location_id", "year_id"])

# Get dengue mortality counts for age specific
as_fhs_df_path = as_fhs_data_path_template.format(
    FHS_DATA_PATH=FHS_DATA_PATH,
    cause_id=cause_id,
    measure_id=6,
    metric_id=1
)
as_fhs_df = read_parquet_with_integer_ids(as_fhs_df_path,
                                     filters=[[year_filter, fhs_location_filter]])
# Rename location_id to fhs_location_id
as_fhs_df = as_fhs_df.rename(columns={
    "location_id": "fhs_location_id",
    "val": "as_fhs_dengue_inc_count"})

df = df.merge(
    as_fhs_df[["fhs_location_id", "year_id",
               "age_group_id","sex_id",
               "as_fhs_dengue_inc_count"]],
    how = "left",
    on = ["fhs_location_id", "year_id", 
          "age_group_id", "sex_id"])

# Get dengue mortality counts for age specific
aa_fhs_df_path = aa_fhs_data_path_template.format(
    FHS_DATA_PATH=FHS_DATA_PATH,
    cause_id=cause_id,
    measure_id=6,
    metric_id=1
)
aa_fhs_df = read_parquet_with_integer_ids(aa_fhs_df_path,
                                     filters=[[year_filter, fhs_location_filter]])
# Rename location_id to fhs_location_id
aa_fhs_df = aa_fhs_df.rename(columns={
    "location_id": "fhs_location_id",
    "val": "aa_fhs_dengue_inc_count"})

df = df.merge(
    aa_fhs_df[["fhs_location_id", "year_id",
                       "aa_fhs_dengue_inc_count"]],
    how = "left",
    on = ["fhs_location_id", "year_id"])


df["as_dengue_inc_fraction"] = df["as_fhs_dengue_inc_count"] / df["aa_fhs_dengue_inc_count"]
df["as_dengue_mort_fraction"] = df["as_fhs_dengue_mort_count"] / df["aa_fhs_dengue_mort_count"]

df["as_dengue_mort_count"] = df["as_dengue_mort_fraction"] * df["aa_dengue_mort_count"]
df["as_dengue_inc_count"] = df["as_dengue_inc_fraction"] * df["aa_dengue_inc_count"]


#
df["as_dengue_inc_rate"] = df["as_dengue_inc_count"] / df["population"]
df["as_dengue_mort_rate"] = df["as_dengue_mort_count"] / df["population"] 

# Drop all columns that have fhs_dengue in them
drop_columns = [col for col in df.columns if "fhs_dengue" in col]
df = df.drop(columns=drop_columns)

df["as_dengue_cfr"] = df["as_dengue_mort_count"] / df["as_dengue_inc_count"]

# Set the CFR to 0 if dengue_inc_count is 0
df.loc[df["as_dengue_inc_count"] == 0, "as_dengue_cfr"] = 0

df.to_parquet(f"{MODELING_DATA_PATH}/as_dengue_stage_2_modeling_df.parquet", index=False)
