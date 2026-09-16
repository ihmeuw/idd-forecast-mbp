"""Build the DAH covariate parquet from the FGH 2026 July program-area DAH CSV.

Reads dah_by_channel_pa_recip_1990_2100.csv (DAH_reference), filters to malaria
program areas (``pa`` codes prefixed ``mal_``) and sums them per (year, recip),
joins to A0 hierarchy + population, and writes per-capita DAH. Finalizes the
_A02_DAH artifact symlink.

DEPENDS ON: _A02_HIERARCHY (read full_hierarchy_2023_*.parquet) and
_A02_POPULATION (read aa_2023_full_population_df.parquet) being current.
The other three economic-variable scripts have no pipeline dependencies.

Importable: `main()` is callable from `00_prep_economic_variables.py`.
Standalone: `python make_dah_df.py` still works (no flags).
"""
import pandas as pd

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.parquet import read_parquet_with_integer_ids, write_parquet
from idd_forecast_mbp.lib.versioning import finish_stage

DAH_SOURCE_PATH = (
    "/mnt/share/resource_tracking/forecasting/dah_channel_HFA/FGH_2026_July"
    "/dah_by_channel_pa_recip_1990_2100.csv"
)


def main() -> None:
    mbpc.DAH_WRITE_PATH.mkdir(parents=True, exist_ok=True)

    hierarchy_df_path = mbpc.HIERARCHY_READ_PATH / f"full_hierarchy_2023_{mbpc.LSAE_HIERARCHY}.parquet"
    aa_full_population_df_path = mbpc.POPULATION_READ_PATH / "aa_2023_full_population_df.parquet"
    dah_df_path = mbpc.DAH_WRITE_PATH / "dah_df.parquet"

    hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)

    A0_hierarchy_df = hierarchy_df[hierarchy_df['level'] == 3].copy()
    A0_hierarchy_df = A0_hierarchy_df[['location_id', 'location_name', 'ihme_loc_id']].drop_duplicates().reset_index(drop=True)
    A0_hierarchy_df = A0_hierarchy_df.rename(columns={'ihme_loc_id': 'iso3'})

    new_dah_df = pd.read_csv(DAH_SOURCE_PATH, usecols=['pa', 'year', 'recip', 'dah_ref'])
    # Malaria = every program area (pa) whose code is prefixed 'mal_' (mal_treat,
    # mal_diag, mal_con_nets, mal_hss_*, ...). This source splits malaria across
    # program areas instead of a single 'mal' health-focus-area, so sum across them.
    new_dah_df = new_dah_df[new_dah_df['pa'].str.startswith('mal_', na=False)
                            & new_dah_df['year'].isin(mbpc.ALL_YEARS)]
    new_dah_df = new_dah_df.groupby(['year', 'recip']).agg({'dah_ref': 'sum'}).reset_index()
    new_dah_df = new_dah_df.rename(columns={'recip': 'iso3', 'dah_ref': 'mal_DAH_total', 'year': 'year_id'})

    new_dah_df = new_dah_df.merge(A0_hierarchy_df, on='iso3', how='inner')

    A0_location_filter = ('location_id', 'in', A0_hierarchy_df['location_id'].unique().tolist())
    pop_df = read_parquet_with_integer_ids(aa_full_population_df_path, filters=[A0_location_filter])

    new_dah_df = new_dah_df.merge(pop_df, on=['location_id', 'year_id'], how='left')
    new_dah_df['mal_DAH_total_per_capita'] = new_dah_df['mal_DAH_total'] / new_dah_df['population']

    write_parquet(new_dah_df, dah_df_path)
    print(f"Wrote {len(new_dah_df):,} rows to {dah_df_path}")
    finish_stage(mbpc._A02_DAH)


if __name__ == "__main__":
    main()
