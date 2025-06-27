from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import read_parquet_with_integer_ids
from idd_forecast_mbp.rake_and_aggregate_functions import make_aa_df_square

measure_map = rfc.measure_map
malaria_variables = rfc.malaria_variables
dengue_variables = rfc.dengue_variables

def format_aa_gbd_df(cause, measure, metric, df, year_start = 2000, year_end = None):
    """
    Formats the GBD DataFrame for a specific cause, measure, and metric.
    
    Parameters:
    - cause: str, the cause name (e.g., 'malaria', 'dengue')
    - measure: str, the measure name (e.g., 'mortality', 'incidence')
    - metric: str, the metric name (e.g., 'count', 'rate')
    - df: DataFrame, the GBD DataFrame to format
    
    Returns:
    - DataFrame with columns: ['location_id', 'year_id', f'{cause}_{measure}_{metric}_count']
    """
    measure_id = rfc.measure_map[measure]['measure_id']
    measure_short = rfc.measure_map[measure]['short']
    metric_id = rfc.metric_map[metric]['metric_id']

    df = df[(df['measure_id'] == measure_id) & (df['metric_id'] == metric_id) & (df['year_id'] >= year_start)].reset_index(drop=True)

    if year_end is not None:
        df = df[df['year_id'] <= year_end]

    df_val_name = f'{cause}_{measure_short}_{metric}'
    df = df.rename(columns={'val': df_val_name})
    df = df[['location_id', 'year_id', df_val_name]]

    return df

def process_lsae_df(cause, measure, aa_full_population_df, hierarchy_df):
    df_path = globals()[f'{cause}_variables'][measure]
    df = read_parquet_with_integer_ids(df_path)
    per_capita_col = [col for col in df.columns if "per_capita" in col][0]
    df = df[["location_id", "year_id", per_capita_col]]
    df = df.merge(aa_full_population_df, on=["location_id", "year_id"], how="left")
    df.loc[df["population"] == 0, per_capita_col] = 0
    if cause == "malaria" and measure == "pfpr":
        new_var_name = "malaria_pfpr"
        df.rename(columns={per_capita_col: new_var_name}, inplace=True)
        # Set all new_var_name values to 0 if population is 0
        df = df[["location_id", "year_id", new_var_name, "population"]]
    elif cause == "malaria":
        measure_short = measure_map[measure]["short"]
        new_var_name = f"{cause}_{measure_short}_count"
        df[new_var_name] = df[per_capita_col] * df["population"]
        df = df[["location_id", "year_id", new_var_name, "population"]]
    else:
        new_var_name = measure
        df.rename(columns={"dengue_suitability_mean_per_capita": new_var_name}, inplace=True)
        df = df[["location_id", "year_id", new_var_name, "population"]]

    df = make_aa_df_square(new_var_name, df, hierarchy_df, level_start = 3, level_end = 5)

    df = df.merge(
        aa_full_population_df[["location_id", "year_id", "population"]],
        on=["location_id", "year_id"],
        how="left",
        suffixes=("", "_full"),
    )
    df["population"] = df["population"].fillna(df["population_full"])
    df = df.drop(columns=["population_full"]) 

    # df = df.merge(hierarchy_df[['location_id', "in_gbd_hierarchy"]], on="location_id", how="left")
    # df = df[df["in_gbd_hierarchy"] == True].drop(columns=["in_gbd_hierarchy"])
    return(df)

def drop_scenario_population_mean(df):
    """
    Drop scenario, population, and any column with the word "mean" in it
    """
    df = rename_columns(df)
    df = df.drop(columns=["scenario", "population"])
    df = df.loc[:, ~df.columns.str.contains("mean")]
    return df

def rename_columns(df):
    """
    Rename columns in the df
    """
    df = df.copy()
    for col in df.columns:
        if "_mean_per_capita" in col:
            df = df.rename(columns={col: col.replace("_mean_per_capita", "")})
        if "rate_mean" in col:
            df = df.rename(columns={col: col.replace("rate_mean", "count")})
    return df