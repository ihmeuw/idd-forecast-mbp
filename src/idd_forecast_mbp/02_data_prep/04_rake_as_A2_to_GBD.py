import numpy as np
import pandas as pd
from pathlib import Path
from rra_tools.shell_tools import mkdir  # type: ignore
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.io.netcdf import convert_to_xarray, write_netcdf
from idd_forecast_mbp.lib.processing.helpers import level_filter
from idd_forecast_mbp.lib.versioning import finalize_artifact


def main(
    lsae_hierarchy: str = mbpc.LSAE_HIERARCHY,
    hierarchy_read_path: Path = mbpc.HIERARCHY_READ_PATH,
    population_read_path: Path = mbpc.POPULATION_READ_PATH,
    mal_raked_aa_read_path: Path = mbpc.MAL_RAKED_AA_READ_PATH,
    den_raked_aa_read_path: Path = mbpc.DEN_RAKED_AA_READ_PATH,
    mal_raked_as_write_path: Path = mbpc.MAL_RAKED_AS_WRITE_PATH,
    den_raked_as_write_path: Path = mbpc.DEN_RAKED_AS_WRITE_PATH,
    gbd_data_path: Path = mbpc.GBD_DATA_PATH,
    causes: list[str] = None,
) -> None:
    if causes is None:
        causes = ['malaria', 'dengue']
    gbd_data_path = Path(gbd_data_path)
    mal_raked_aa_read_path = Path(mal_raked_aa_read_path)
    den_raked_aa_read_path = Path(den_raked_aa_read_path)
    mal_raked_as_write_path = Path(mal_raked_as_write_path)
    den_raked_as_write_path = Path(den_raked_as_write_path)

    # Write paths (per cause)
    mal_raked_as_write_path.mkdir(parents=True, exist_ok=True)
    den_raked_as_write_path.mkdir(parents=True, exist_ok=True)
    as_write_paths = {
        'malaria':     mal_raked_as_write_path,
        'malaria_pf':  mal_raked_as_write_path,
        'malaria_pv':  mal_raked_as_write_path,
        'dengue':      den_raked_as_write_path,
    }

    # Read paths for aa raked outputs (per cause)
    aa_raked_read_paths = {
        'malaria':     mal_raked_aa_read_path,
        'malaria_pf':  mal_raked_aa_read_path,
        'malaria_pv':  mal_raked_aa_read_path,
        'dengue':      den_raked_aa_read_path,
    }

    ################################################################
    #### Paths, loading, and cleaning
    ################################################################

    aa_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_aa.parquet"
    as_gbd_cause_df_path_template = "{GBD_DATA_PATH}/gbd_2023_{cause}_as.parquet"

    years = list(range(2000, 2023))
    year_filter = ('year_id', 'in', years)

    as_full_population_df_path = Path(population_read_path) / "as_2023_full_population_df.parquet"
    as_full_population_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                    filters = [year_filter]).drop(columns=['as_population_fraction'])

    full_2023_hierarchy_path = Path(hierarchy_read_path) / f"full_hierarchy_2023_{lsae_hierarchy}.parquet"
    age_sex_df_path = Path(population_read_path) / "age_sex_df.parquet"

    hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)
    age_sex_df = read_parquet_with_integer_ids(age_sex_df_path)
    #################################################################
    #### Constants
    #################################################################
    cause_map = mbpc.cause_map
    measure_map = mbpc.measure_map
    metric_map = mbpc.metric_map

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
    for cause in causes:
        print(f"Starting {cause} {metric}")
        as_full_dfs = []
        as_gbd_cause_df_path = as_gbd_cause_df_path_template.format(GBD_DATA_PATH=gbd_data_path, cause=cause)
        aa_full_cause_df_path = aa_raked_read_paths[cause] / f"aa_full_{cause}_df.parquet"
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
        as_full_cause_df_path = as_write_paths[cause] / f"as_full_{cause}_df.parquet"
        as_full_cause_ds_path = as_write_paths[cause] / f"as_full_{cause}_ds.nc"
        write_parquet(as_full_cause_df, as_full_cause_df_path)


        as_full_cause_ds = convert_to_xarray(
            as_full_cause_df,
            dimensions=['location_id', 'year_id', 'sex_id', 'age_group_id'],
            dimension_dtypes={'location_id': 'int32', 'year_id': 'int16', 'sex_id': 'int16', 'age_group_id': 'int16'},
            auto_optimize_dtypes=True
        )

        write_netcdf(as_full_cause_ds, as_full_cause_ds_path,
            compression=True,
            compression_level=4,
            chunking=True,
            chunk_by_dim={'location_id': 1500, 'year_id': 79},
            engine='netcdf4'
        )
        print(f"Wrote {as_full_cause_df_path}")

    if 'malaria' in causes:
        finalize_artifact(mbpc._A02_MAL_RAKED_AS)
    if 'dengue' in causes:
        finalize_artifact(mbpc._A02_DEN_RAKED_AS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rake age-sex A2 counts to GBD")
    parser.add_argument("--causes", nargs="+", default=None,
                        help="Causes to process (e.g. --causes malaria). Defaults to malaria and dengue.")
    args = parser.parse_args()
    main(causes=args.causes)
