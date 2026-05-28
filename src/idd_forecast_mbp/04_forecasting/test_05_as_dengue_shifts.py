import pandas as pd
import numpy as np
from pathlib import Path
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids
from idd_forecast_mbp.lib.io.netcdf import convert_with_preset, write_netcdf, read_netcdf_with_integer_ids
from idd_forecast_mbp.lib.processing.disaggregation import disaggregate_age_sex_dengue

TEST_DIR = Path("/mnt/team/idd/pub/forecast-mbp/test_output/04-forecasting_data")

ssp_scenario = "ssp126"
draw = "001"
hold_variable = "None"
cause = "dengue"
modeling_measure = "incidence"

PROCESSED_DATA_PATH = mbpc.PROCESSED_DATA_PATH
FORECASTING_DATA_PATH = mbpc.FORECASTING_DATA_PATH

# Paths — reads from production raked file
input_cause_draw_path = f"{FORECASTING_DATA_PATH}/raked_{cause}_forecast_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
output_dengue_incidence_draw_path = TEST_DIR / f"as_{cause}_measure_incidence_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"
output_dengue_mortality_draw_path = TEST_DIR / f"as_{cause}_measure_mortality_ssp_scenario_{ssp_scenario}_draw_{draw}_with_predictions.nc"

as_full_population_df_path = f"{PROCESSED_DATA_PATH}/as_2023_full_population.parquet"
full_2023_hierarchy_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_lsae_1209.parquet"
age_sex_df_path = f"{PROCESSED_DATA_PATH}/age_sex_df.parquet"

hierarchy_df = read_parquet_with_integer_ids(full_2023_hierarchy_path)
age_sex_df = read_parquet_with_integer_ids(age_sex_df_path)

cause_map = mbpc.cause_map
modeling_measure_map = mbpc.modeling_measure_map
reference_age_group_id = cause_map[cause]['reference_age_group_id']
reference_sex_id = cause_map[cause]['reference_sex_id']

as_merge_variables = mbpc.as_merge_variables

columns_to_read = as_merge_variables + ['logit_dengue_cfr_pred', 'base_log_dengue_inc_rate_pred']
future_year_ids = list(range(2022, 2101))
year_filter = ('year_id', 'in', future_year_ids)

ds = read_netcdf_with_integer_ids(input_cause_draw_path)
ds = ds.sel(year_id=future_year_ids)
df = ds.to_dataframe().reset_index()
df = df[columns_to_read].copy()

df['dengue_cfr_pred'] = 1 / (1 + np.exp(-df['logit_dengue_cfr_pred']))
df = df.drop(columns=['logit_dengue_cfr_pred'])

df_location_ids = df['location_id'].unique().tolist()
df_location_filter = ('location_id', 'in', df_location_ids)

pop_cols_to_read = as_merge_variables + ['population']
as_population_df = read_parquet_with_integer_ids(as_full_population_df_path,
                                                 columns=pop_cols_to_read,
                                                 filters=[df_location_filter, year_filter])
as_population_df = as_population_df.merge(hierarchy_df[['location_id', 'gbd_location_id']], on='location_id', how='left')

forecast_df = as_population_df.merge(df, on=as_merge_variables, how='left').copy()

as_md_gbd_dengue_df_path = f"{PROCESSED_DATA_PATH}/as_md_gbd_dengue_df.parquet"
columns_to_read = ['location_id', 'sex_id', 'age_group_id', 'rr_inc_as']
as_md_gbd_dengue_df = read_parquet_with_integer_ids(as_md_gbd_dengue_df_path,
                                                    columns=columns_to_read).rename(columns={
    'location_id': 'gbd_location_id'
})

forecast_df = forecast_df.merge(as_md_gbd_dengue_df, on=['gbd_location_id', 'age_group_id', 'sex_id'], how='left')

forecast_df['base_log_dengue_inc_rate_pred'] = forecast_df['base_log_dengue_inc_rate_pred'].fillna(0)
forecast_df['dengue_cfr_pred'] = forecast_df['dengue_cfr_pred'].fillna(0)

forecast_df = disaggregate_age_sex_dengue(forecast_df)

keep_columns = as_merge_variables + ['population', 'dengue_inc_count_pred', 'dengue_mort_count_pred']
forecast_df = forecast_df[keep_columns]

non_measure_columns = [col for col in forecast_df.columns if 'inc' not in col and 'mort' not in col]
incidence_columns = [col for col in forecast_df.columns if 'inc' in col]
mortality_columns = [col for col in forecast_df.columns if 'mort' in col]

# Apply vaccination effects (Singapore, Brazil, Indonesia, Thailand children)
locations = ['Singapore', 'Brazil', 'Indonesia', 'Thailand']
location_ids = hierarchy_df[hierarchy_df['location_name'].isin(locations)]['location_id'].unique()
children_ids = hierarchy_df[hierarchy_df['parent_id'].isin(location_ids)]['location_id'].unique()
grand_children_ids = hierarchy_df[hierarchy_df['parent_id'].isin(children_ids)]['location_id'].unique()

dengue_vaccine_df_path = f"{FORECASTING_DATA_PATH}/dengue_vaccine_df.parquet"
dengue_vaccine_df = read_parquet_with_integer_ids(dengue_vaccine_df_path)
vaccine_lookup = dengue_vaccine_df.set_index('age_group_id')

vaccine_mask = (
    forecast_df['location_id'].isin(grand_children_ids) &
    (forecast_df['year_id'] >= 2023) &
    (forecast_df['year_id'] <= 2100)
)

for year in range(2023, 2101):
    year_col = f'year_{year}'
    if year_col in vaccine_lookup.columns:
        year_mask = vaccine_mask & (forecast_df['year_id'] == year)
        if year_mask.any():
            age_group_reductions = vaccine_lookup[year_col]
            reductions = forecast_df.loc[year_mask, 'age_group_id'].map(age_group_reductions)
            reductions = reductions.fillna(1.0)
            forecast_df.loc[year_mask, 'dengue_mort_count_pred'] *= reductions

mortality_df = forecast_df[non_measure_columns + mortality_columns]
mortality_ds = convert_with_preset(mortality_df, preset='as_variables')
write_netcdf(mortality_ds, output_dengue_mortality_draw_path)
print(f"Wrote {output_dengue_mortality_draw_path}")

incidence_df = forecast_df[non_measure_columns + incidence_columns]
incidence_ds = convert_with_preset(incidence_df, preset='as_variables')
write_netcdf(incidence_ds, output_dengue_incidence_draw_path)
print(f"Wrote {output_dengue_incidence_draw_path}")
