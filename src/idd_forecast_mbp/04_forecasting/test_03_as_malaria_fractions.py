import pandas as pd
import numpy as np
import gc
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.netcdf import convert_with_preset, write_netcdf, read_netcdf_with_integer_ids
from idd_forecast_mbp.lib.processing.disaggregation import disaggregate_age_sex_malaria

TEST_DIR = Path("/mnt/team/idd/pub/forecast-mbp/test_output/04-forecasting_data")

ssp_scenario = "ssp126"
dah_scenario = "Baseline"
draw = "001"
hold_variable = "None"
cause = "malaria"

PROCESSED_DATA_PATH = mbpc.PROCESSED_DATA_PATH
FORECASTING_DATA_PATH = mbpc.FORECASTING_DATA_PATH

# Paths
input_cause_draw_path = f"{FORECASTING_DATA_PATH}/{cause}_forecast_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
output_malaria_incidence_draw_path = TEST_DIR / f"as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"
output_malaria_mortality_draw_path = TEST_DIR / f"as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_dah_scenario_{dah_scenario}_draw_{draw}_with_predictions.nc"

aa_full_population_df_path = f"{PROCESSED_DATA_PATH}/aa_2023_full_population.parquet"
as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
full_2023_hierarchy_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet"
age_sex_df_path = f"{PROCESSED_DATA_PATH}/age_sex_df.parquet"
as_md_gbd_malaria_df_path = f"{PROCESSED_DATA_PATH}/as_md_gbd_malaria_df.parquet"

hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)
age_sex_df = read_parquet_with_integer_ids(age_sex_df_path)

cause_map = mbpc.cause_map
reference_age_group_id = cause_map[cause]['reference_age_group_id']
reference_sex_id = cause_map[cause]['reference_sex_id']

as_merge_variables = mbpc.as_merge_variables

forecast_years = list(range(2022, 2101))
forecast_year_filter = ('year_id', 'in', forecast_years)

ds = read_netcdf_with_integer_ids(input_cause_draw_path)
ds = ds.sel(year_id=forecast_years)
df = ds.to_dataframe().reset_index()
modeling_location_ids = df['location_id'].unique().tolist()
modeling_location_filter = ('location_id', 'in', modeling_location_ids)

aa_pop_df = read_parquet_with_integer_ids(aa_full_population_df_path,
                                         filters=[modeling_location_filter, forecast_year_filter])
aa_pop_df = aa_pop_df.rename(columns={'population': 'aa_population'})
df = df.merge(aa_pop_df[['location_id', 'year_id', 'aa_population']],
              on=['location_id', 'year_id'], how='left')

df['aa_malaria_mort_rate'] = np.exp(df['log_aa_malaria_mort_rate_pred'])
df['aa_malaria_inc_rate'] = np.exp(df['log_aa_malaria_inc_rate_pred'])
df = df.drop(columns=['log_aa_malaria_mort_rate_pred', 'log_aa_malaria_inc_rate_pred'])
for col in df.columns:
    if 'malaria' in col:
        df[col] = df[col].fillna(0)

df['aa_malaria_mort_count'] = df['aa_malaria_mort_rate'] * df['aa_population']
df['aa_malaria_inc_count'] = df['aa_malaria_inc_rate'] * df['aa_population']

md_gbd_location_df = hierarchy_df[hierarchy_df['most_detailed_gbd'] == True].copy()
last_year = min(forecast_years)
md_gbd_location_filter = ('location_id', 'in', md_gbd_location_df['location_id'].unique().tolist())

as_md_gbd_malaria_df = read_parquet_with_integer_ids(as_md_gbd_malaria_df_path)
as_md_gbd_malaria_df = as_md_gbd_malaria_df.rename(columns={'location_id': 'gbd_location_id'})
as_md_gbd_malaria_df = as_md_gbd_malaria_df[(as_md_gbd_malaria_df['rr_inc_as'] > 0) | (as_md_gbd_malaria_df['rr_mort_as'] > 0)].copy()
gbd_location_ids = as_md_gbd_malaria_df['gbd_location_id'].unique().tolist()

level_5_location_ids = hierarchy_df[(hierarchy_df['level'] == 5) & (hierarchy_df['gbd_location_id'].isin(gbd_location_ids))]['location_id'].unique().tolist()
level_5_location_filter = ('location_id', 'in', level_5_location_ids)

forecast_columns_to_read = as_merge_variables + ['population']
forecast_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                            columns=forecast_columns_to_read,
                                            filters=[level_5_location_filter, forecast_year_filter])

forecast_df = forecast_df.merge(df[['location_id', 'year_id', 'aa_malaria_mort_count', 'aa_malaria_inc_count']],
                                on=['location_id', 'year_id'], how='left')
del df
gc.collect()

forecast_df = forecast_df.merge(hierarchy_df[['location_id', 'gbd_location_id']],
                                on='location_id', how='left')
forecast_df = forecast_df.merge(as_md_gbd_malaria_df[['gbd_location_id', 'age_group_id', 'sex_id', 'rr_inc_as', 'rr_mort_as']],
                                on=['gbd_location_id', 'age_group_id', 'sex_id'], how='left')
del as_md_gbd_malaria_df
gc.collect()

forecast_df = disaggregate_age_sex_malaria(forecast_df)
forecast_df = forecast_df.drop(columns=['rr_inc_as', 'rr_mort_as', 'population'])

non_measure_columns = [col for col in forecast_df.columns if 'inc' not in col and 'mort' not in col]
incidence_columns = [col for col in forecast_df.columns if 'inc' in col]
mortality_columns = [col for col in forecast_df.columns if 'mort' in col]

incidence_df = forecast_df[non_measure_columns + incidence_columns]
incidence_ds = convert_with_preset(incidence_df, preset='as_variables')
write_netcdf(incidence_ds, output_malaria_incidence_draw_path)
print(f"Wrote {output_malaria_incidence_draw_path}")

mortality_df = forecast_df[non_measure_columns + mortality_columns]
mortality_ds = convert_with_preset(mortality_df, preset='as_variables')
write_netcdf(mortality_ds, output_malaria_mortality_draw_path)
print(f"Wrote {output_malaria_mortality_draw_path}")

del forecast_df
gc.collect()
