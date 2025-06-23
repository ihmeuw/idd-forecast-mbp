import numpy as np
import pandas as pd
from pathlib import Path
from rra_tools.shell_tools import mkdir  # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import read_parquet_with_integer_ids, write_parquet, level_filter


PROCESSED_DATA_PATH = rfc.PROCESSED_DATA_PATH

GBD_DATA_PATH = rfc.GBD_DATA_PATH
FHS_DATA_PATH = f"{PROCESSED_DATA_PATH}/age_specific_fhs"

as_full_cause_df_path_template = '{PROCESSED_DATA_PATH}/as_full_{cause}_df.parquet'
################################################################
#### Paths, loading, and cleaning
################################################################

aa_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_aa.parquet"
as_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_as.parquet"
aa_full_cause_df_path_template = '{PROCESSED_DATA_PATH}/aa_full_{cause}_df.parquet'

as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
as_full_population_df = read_parquet_with_integer_ids(as_full_population_df_path).drop(columns=['fhs_location_id'])

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
years = list(range(2000, 2023))
year_filter = ('year_id', 'in', years)
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
    as_full_dfs = []

    as_gbd_cause_df_path = as_gbd_cause_df_path_template.format(GBD_DATA_PATH=GBD_DATA_PATH, cause=cause)
    aa_full_cause_df_path = aa_full_cause_df_path_template.format(PROCESSED_DATA_PATH=PROCESSED_DATA_PATH, cause=cause)
    for measure in measure_map:
        measure_short = measure_map[measure]['short']
        measure_filter = ('measure_id', '==', measure_map[measure]['measure_id'])
        metric_filter = ('metric_id', '==', metric_map[metric]['metric_id'])

        outcome = f'{cause}_{measure_short}_{metric}'
        gbd_columns_to_read = ["location_id", "year_id", "age_group_id", "sex_id", "val"]
        full_df_columns_to_read = ["location_id", "year_id", outcome]
        # Get the most detailed gbd data
        as_md_gbd_df = read_parquet_with_integer_ids(as_gbd_cause_df_path,
                                                    columns=gbd_columns_to_read,
                                                    filters=[year_filter, level_filter(hierarchy_df, start_level = 3, end_level = 5), measure_filter,
                                                            metric_filter, age_filter, sex_filter]).rename(columns={'val': outcome})
        gbd_location_ids = as_md_gbd_df['location_id'].unique().tolist()
        gbd_location_filter = ('location_id', 'in', gbd_location_ids)
        aa_md_gbd_df = read_parquet_with_integer_ids(aa_full_cause_df_path,
                                                    columns=full_df_columns_to_read,
                                                    filters=[year_filter, gbd_location_filter]).rename(columns={outcome: 'aa_' + outcome})

        as_md_gbd_df = as_md_gbd_df.merge(aa_md_gbd_df, on=['location_id', 'year_id'], how='left').copy()
        as_md_gbd_df['as_to_aa_' + outcome + '_fraction'] = as_md_gbd_df[outcome] / as_md_gbd_df['aa_' + outcome]
        # Set the fraction to 0 where as_md_gbd_df['aa_' + outcome] is 0
        as_md_gbd_df.loc[as_md_gbd_df['aa_' + outcome] == 0, 'as_to_aa_' + outcome + '_fraction'] = 0
        as_md_gbd_df['set_by_gbd'] = True

        # Create the subnational as dataframe
        aa_subnat_df = read_parquet_with_integer_ids(aa_full_cause_df_path,
                                                    columns=full_df_columns_to_read,
                                                    filters=[year_filter, level_filter(hierarchy_df, start_level = 4, end_level = 5)])
        aa_subnat_df = aa_subnat_df.rename(columns={outcome: 'aa_' + outcome})

        # Created as_level dataframe
        as_subnat_df = aa_subnat_df.merge(age_sex_df, how = "cross")[as_merge_variables + ['aa_' + outcome]]

        # Merge in subnat GBD and set up the mask so we don't overwrite GBD values
        as_subnat_df = as_subnat_df.merge(
            as_md_gbd_df[as_merge_variables + [outcome, 'set_by_gbd']],
            how='left',
            on=as_merge_variables
            )

        # Track which locations are already set by GBD
        mask = as_subnat_df['set_by_gbd'].isna()
        as_subnat_df.loc[mask, 'set_by_gbd'] = False
        as_subnat_df['set_by_gbd'] = as_subnat_df['set_by_gbd'].astype('boolean')

        # Merge in the gbd location ids)
        as_subnat_df = as_subnat_df.merge(
            hierarchy_df[['location_id', 'gbd_location_id']],
            how='left', on='location_id'
        )

        # Rename the gbd dataframe columns
        gbd_outcome_columns = [col for col in as_md_gbd_df.columns if outcome in col] + ['location_id']
        rename_dict = {col: 'gbd_' + col for col in gbd_outcome_columns}
        as_md_gbd_df = as_md_gbd_df.rename(columns=rename_dict).drop(columns=['set_by_gbd'])

        # Merge again
        as_subnat_df = as_subnat_df.merge(
            as_md_gbd_df,
            how='left',
            on=['gbd_location_id', 'year_id', 'age_group_id', 'sex_id']
        )
        mask = as_subnat_df['set_by_gbd'] == False
        as_subnat_df.loc[mask, outcome] = (
            as_subnat_df.loc[mask, 'aa_' + outcome] * 
            as_subnat_df.loc[mask, 'gbd_as_to_aa_' + outcome + '_fraction']
        )

        columns_to_keep = as_merge_variables + [outcome, 'aa_' + outcome]
        as_subnat_df = as_subnat_df[columns_to_keep].copy()

        # Get the rest of the hierarchy
        as_rest_df = read_parquet_with_integer_ids(as_gbd_cause_df_path,
                                                columns=gbd_columns_to_read,
                                                filters=[year_filter, level_filter(hierarchy_df, start_level = 0, end_level = 3), measure_filter,
                                                            metric_filter, age_filter, sex_filter]).rename(columns={'val': outcome})
        gbd_location_ids = as_rest_df['location_id'].unique().tolist()
        gbd_location_filter = ('location_id', 'in', gbd_location_ids)
        aa_rest_df = read_parquet_with_integer_ids(aa_full_cause_df_path,
                                                columns=full_df_columns_to_read,
                                                filters=[year_filter, gbd_location_filter]).rename(columns={outcome: 'aa_' + outcome})
        as_rest_df = as_rest_df.merge(aa_rest_df, on=['location_id', 'year_id'], how='left').copy()

        as_full_df = pd.concat([as_subnat_df, as_rest_df], ignore_index=True)
        as_full_dfs.append(as_full_df)
    # Merge the two measure dfs together
    as_full_cause_df = as_full_dfs[0].copy()
    as_full_cause_df = as_full_cause_df.merge(as_full_dfs[1], on=as_merge_variables, how='left').copy()
    as_full_cause_df = as_full_cause_df.merge(as_full_population_df, on=as_merge_variables, how='left').copy()
    force_zero_age_ids = force_zero.get(cause, [])
    if len(force_zero_age_ids) > 0:
        as_full_cause_df.loc[as_full_cause_df['age_group_id'].isin(force_zero_age_ids), [f'{cause}_{measure_map[measure]["short"]}_{metric}' for measure in measure_map]] = 0
        as_full_cause_df.loc[as_full_cause_df['age_group_id'].isin(force_zero_age_ids), [f'aa_{cause}_{measure_map[measure]["short"]}_{metric}' for measure in measure_map]] = 0
    for measure in measure_map:
        measure_short = measure_map[measure]['short']
        count_outcome = f'{cause}_{measure_short}_{metric}'
        rate_outcome = f'{cause}_{measure_short}_rate'
        as_full_cause_df[rate_outcome] = as_full_cause_df[count_outcome] / as_full_cause_df['population']
        as_full_cause_df.loc[as_full_cause_df['population'] == 0, rate_outcome] = 0
        aa_rate_outcome = f'aa_{cause}_{measure_short}_rate'
        as_full_cause_df[aa_rate_outcome] = as_full_cause_df[f'aa_{cause}_{measure_short}_{metric}'] / as_full_cause_df['aa_population']
        as_full_cause_df.loc[as_full_cause_df['aa_population'] == 0, aa_rate_outcome] = 0
    as_full_cause_df_path = as_full_cause_df_path_template.format(PROCESSED_DATA_PATH=PROCESSED_DATA_PATH, cause=cause)
    write_parquet(as_full_cause_df, as_full_cause_df_path)
    print(f"Wrote {as_full_cause_df_path}")