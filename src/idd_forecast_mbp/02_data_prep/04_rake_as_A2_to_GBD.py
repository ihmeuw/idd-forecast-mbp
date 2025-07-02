import numpy as np
import pandas as pd
from pathlib import Path
from rra_tools.shell_tools import mkdir  # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import level_filter
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet

PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH
FORECASTING_DATA_PATH = rfc.FORECASTING_DATA_PATH
GBD_DATA_PATH = rfc.GBD_DATA_PATH
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"

as_full_cause_df_path_template = '{PROCESSED_DATA_PATH}/as_full_{cause}_df.parquet'
################################################################
#### Paths, loading, and cleaning
################################################################

aa_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_aa.parquet"
as_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_as.parquet"
aa_full_cause_df_path_template = '{PROCESSED_DATA_PATH}/aa_full_{cause}_df.parquet'

years = list(range(2000, 2023))
year_filter = ('year_id', 'in', years)

as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
as_full_population_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                filters = [year_filter]).drop(columns=['as_population_fraction'])

full_2023_hierarchy_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet"
age_sex_df_path = f'{PROCESSED_DATA_PATH}/age_sex_df.parquet'

hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)
age_sex_df = read_parquet_with_integer_ids(age_sex_df_path)
#################################################################
#### Constants
#################################################################
cause_map = rfc.cause_map
measure_map = rfc.measure_map
metric_map = rfc.metric_map

sex_ids = [1, 2]
sex_filter = ('sex_id', 'in', sex_ids)
as_merge_variables = ["location_id", "year_id", "age_group_id", "sex_id"]
age_group_ids = age_sex_df['age_group_id'].unique().tolist()
age_filter = ('age_group_id', 'in', age_group_ids)

force_zero = {
    'malaria': [2],
    'dengue': [2]
}

metric = "count"
for cause in cause_map:
    print(f"Starting {cause} {metric}")
    as_full_dfs = []
    as_gbd_cause_df_path = as_gbd_cause_df_path_template.format(GBD_DATA_PATH=GBD_DATA_PATH, cause=cause)
    aa_full_cause_df_path = aa_full_cause_df_path_template.format(PROCESSED_DATA_PATH=PROCESSED_DATA_PATH, cause=cause)
    for measure in measure_map:
        print(f"Starting {cause} {measure}")
        measure_short = measure_map[measure]['short']
        measure_filter = ('measure_id', '==', measure_map[measure]['measure_id'])
        metric_filter = ('metric_id', '==', metric_map[metric]['metric_id'])

        outcome_count = f'{cause}_{measure_short}_{metric}'
        outcome_rate = f'{cause}_{measure_short}_rate'

        gbd_columns_to_read = ["location_id", "year_id", "age_group_id", "sex_id", 'population', "val"]
        full_df_columns_to_read = ["location_id", "year_id", 'population', outcome_count]
        # Get the most detailed gbd data
        print("Reading most-detailed GBD data")
        print(f"Reading {as_gbd_cause_df_path}")
        as_md_gbd_df = read_parquet_with_integer_ids(as_gbd_cause_df_path,
                                                    columns=gbd_columns_to_read,
                                                    filters=[year_filter, level_filter(hierarchy_df, start_level = 3, end_level = 5), measure_filter,
                                                            metric_filter, age_filter, sex_filter]).rename(columns={'val': outcome_count})

        gbd_location_ids = as_md_gbd_df['location_id'].unique().tolist()
        gbd_location_filter = ('location_id', 'in', gbd_location_ids)
        print(f"Reading {aa_full_cause_df_path}")
        aa_md_gbd_df = read_parquet_with_integer_ids(aa_full_cause_df_path,
                                                    columns=full_df_columns_to_read,
                                                    filters=[year_filter, gbd_location_filter]).rename(columns={
                                                        outcome_count: 'aa_' + outcome_count,
                                                        'population': 'aa_population'})

        as_md_gbd_df = as_md_gbd_df.merge(aa_md_gbd_df, on=['location_id', 'year_id'], how='left').copy()

        print("Calculating most-detailed GBD rates and rate ratios")
        as_md_gbd_df['aa_' + outcome_rate] = as_md_gbd_df['aa_' + outcome_count] / as_md_gbd_df['aa_population']
        as_md_gbd_df[outcome_rate] = as_md_gbd_df[outcome_count] / as_md_gbd_df['population']
        as_md_gbd_df['rate_ratio'] = as_md_gbd_df[outcome_rate] / as_md_gbd_df['aa_' + outcome_rate]
        as_md_gbd_df.loc[as_md_gbd_df['aa_' + outcome_rate] == 0, 'rate_ratio'] = 0

        # Create the subnational as dataframe
        aa_subnat_df = read_parquet_with_integer_ids(aa_full_cause_df_path,
                                                    columns=full_df_columns_to_read,
                                                    filters=[year_filter, level_filter(hierarchy_df, start_level = 4, end_level = 5)])
        aa_subnat_df = aa_subnat_df.rename(columns={outcome_count: 'aa_' + outcome_count})

        # Created as_level dataframe
        as_subnat_df = aa_subnat_df.merge(age_sex_df, how = "cross")[as_merge_variables + ['aa_' + outcome_count]]

        as_subnat_df = as_subnat_df.merge(as_full_population_df, on=as_merge_variables, how='left')

        # Merge in the gbd location ids
        as_subnat_df = as_subnat_df.merge(
            hierarchy_df[['location_id', 'gbd_location_id']],
            how='left', on='location_id'
        )

        # Rename the gbd dataframe columns
        gbd_outcome_columns = [col for col in as_md_gbd_df.columns if measure_short in col or 'ratio' in col] + ['location_id','aa_population', 'population']
        rename_dict = {col: 'gbd_' + col for col in gbd_outcome_columns}
        as_md_gbd_df = as_md_gbd_df.rename(columns=rename_dict)

        print("Starting subnational merge")
        # Merge
        as_subnat_df = as_subnat_df.merge(
            as_md_gbd_df,
            how='left',
            on=['gbd_location_id', 'year_id', 'age_group_id', 'sex_id']
        )
        
        
        as_subnat_df['aa_' + outcome_rate] = as_subnat_df['aa_' + outcome_count] / as_subnat_df['aa_population']
        as_subnat_df[outcome_rate] = as_subnat_df['gbd_rate_ratio'] * as_subnat_df['aa_' + outcome_rate]
        as_subnat_df[outcome_count] = as_subnat_df[outcome_rate] * as_subnat_df['population']
        # 
        print(as_subnat_df[as_subnat_df[outcome_rate] == as_subnat_df[outcome_rate].max()])
        drop_cols = [col for col in as_subnat_df.columns if 'gbd_' in col or 'rate_ratio' in col]
        as_subnat_df = as_subnat_df.drop(columns=drop_cols)

        as_rest_df = read_parquet_with_integer_ids(as_gbd_cause_df_path,
                                                columns=gbd_columns_to_read,
                                                filters=[year_filter, level_filter(hierarchy_df, start_level = 0, end_level = 3), measure_filter,
                                                                metric_filter, age_filter, sex_filter]).rename(columns={'val': outcome_count})
        gbd_location_ids = as_rest_df['location_id'].unique().tolist()
        gbd_location_filter = ('location_id', 'in', gbd_location_ids)
        aa_rest_df = read_parquet_with_integer_ids(aa_full_cause_df_path,
                                                columns=full_df_columns_to_read,
                                                filters=[year_filter, gbd_location_filter]).rename(columns={outcome_count: 'aa_' + outcome_count, 'population': 'aa_population'})
        as_rest_df = as_rest_df.merge(aa_rest_df, on=['location_id', 'year_id'], how='left').copy()
        as_rest_df['aa_' + outcome_rate] = as_rest_df['aa_' + outcome_count] / as_rest_df['aa_population']
        as_rest_df[outcome_rate] = as_rest_df[outcome_count] / as_rest_df['population']

        as_full_df = pd.concat([as_subnat_df, as_rest_df], ignore_index=True)
        as_full_dfs.append(as_full_df)

    print('Starting final merge')
    as_full_cause_df = as_full_dfs[0].copy()
    merge_cols = [col for col in as_full_dfs[1].columns if 'pop' not in col]
    as_full_cause_df = as_full_cause_df.merge(as_full_dfs[1][merge_cols], on=as_merge_variables, how='left')
    force_zero_age_ids = force_zero.get(cause, [])
        
    if len(force_zero_age_ids) > 0:
        as_full_cause_df.loc[as_full_cause_df['age_group_id'].isin(force_zero_age_ids), [f'{cause}_{measure_map[measure]["short"]}_{metric}' for measure in measure_map]] = 0
        as_full_cause_df.loc[as_full_cause_df['age_group_id'].isin(force_zero_age_ids), [f'aa_{cause}_{measure_map[measure]["short"]}_{metric}' for measure in measure_map]] = 0
    as_full_cause_df_path = as_full_cause_df_path_template.format(PROCESSED_DATA_PATH=PROCESSED_DATA_PATH, cause=cause)
    write_parquet(as_full_cause_df, as_full_cause_df_path)
    print(f"Wrote {as_full_cause_df_path}")