import pandas as pd
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.versioning import finalize_artifact

mbpc.DAH_WRITE_PATH.mkdir(parents=True, exist_ok=True)

hierarchy_df_path = mbpc.HIERARCHY_READ_PATH / f"full_hierarchy_2023_{mbpc.LSAE_HIERARCHY}.parquet"
aa_full_population_df_path = mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet"
dah_df_path = mbpc.DAH_WRITE_PATH / "dah_df.parquet"

new_dah_path = '/mnt/share/resource_tracking/forecasting/dah_channel_HFA/FGH_2026_Feb/dah_by_channel_hfa_recip_1990_2100.csv'

hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

A0_hierarchy_df = hierarchy_df[hierarchy_df['level'] == 3].copy()
A0_hierarchy_df = A0_hierarchy_df[['location_id', 'location_name', 'ihme_loc_id']].drop_duplicates().reset_index(drop=True)
A0_hierarchy_df = A0_hierarchy_df.rename(columns={'ihme_loc_id': 'iso3'})

new_dah_df = pd.read_csv(new_dah_path)
new_dah_df = new_dah_df[(new_dah_df['hfa'] == 'mal') & (new_dah_df['year'] >= 2000)]
new_dah_df = new_dah_df.groupby(['year', 'recip']).agg({'dah_ref': 'sum'}).reset_index()
new_dah_df = new_dah_df.rename(columns={'recip': 'iso3', 'dah_ref': 'mal_DAH_total', 'year': 'year_id'})

new_dah_df = new_dah_df.merge(A0_hierarchy_df, on='iso3', how='inner')

A0_location_filter = ('location_id', 'in', A0_hierarchy_df['location_id'].unique().tolist())
pop_df = read_parquet_with_integer_ids(aa_full_population_df_path, filters=[A0_location_filter])

new_dah_df = new_dah_df.merge(pop_df, on=['location_id', 'year_id'], how='left')
new_dah_df['mal_DAH_total_per_capita'] = new_dah_df['mal_DAH_total'] / new_dah_df['population']

write_parquet(new_dah_df, dah_df_path)
finalize_artifact(mbpc._A02_DAH)
