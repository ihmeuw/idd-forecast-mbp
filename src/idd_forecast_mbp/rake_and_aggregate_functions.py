import pandas as pd
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import write_parquet

def check_concordance(variable, aa_full_df, aa_gbd_df, tolerance = 0.01):
    aa_gbd_df = aa_gbd_df.rename(columns={variable: f'gbd_{variable}'})
    combined_df = pd.merge(aa_full_df, aa_gbd_df, on=['location_id', 'year_id'], how='inner')
    combined_df['concordance'] = (combined_df[variable] - combined_df[f'gbd_{variable}']).abs()
    # Put combined_df in order based on concordance descending
    combined_df = combined_df.sort_values(by='concordance', ascending=False).reset_index(drop=True)
    concordance_stats = {}
    if combined_df['concordance'].max() > tolerance:
        print(f'Warning: Maximum absolute difference for {variable} exceeds tolerance of {tolerance}: {combined_df["concordance"].max()}')
        concordance_stats = {
            'mean_absolute_diff': combined_df['concordance'].mean(),
            'median_absolute_diff': combined_df['concordance'].median(),
            'max_absolute_diff': combined_df['concordance'].max(),
            'std_absolute_diff': combined_df['concordance'].std(),
            'mean_relative_diff_pct': (combined_df['concordance'] / combined_df[f'gbd_{variable}']).mean() * 100,
            'pearson_correlation': combined_df[variable].corr(combined_df[f'gbd_{variable}']),
            'within_1_pct': (combined_df['concordance'] / combined_df[f'gbd_{variable}'] <= 0.01).mean(),
            'within_5_pct': (combined_df['concordance'] / combined_df[f'gbd_{variable}'] <= 0.05).mean(),
            'within_10_pct': (combined_df['concordance'] / combined_df[f'gbd_{variable}'] <= 0.10).mean(),
            'p95_absolute_diff': combined_df['concordance'].quantile(0.95),
            'p99_absolute_diff': combined_df['concordance'].quantile(0.99)
        }
        print(combined_df.head(10))
    else:
        print(f'✅ {variable} concordance check passed: max absolute difference {combined_df["concordance"].max()} within tolerance {tolerance}')
        concordance_stats = {}
    return concordance_stats  


def make_aa_df_square(variable, df, hierarchy_df, level_start, level_end):
    df = df.copy()
    years = df['year_id'].unique()
    level_hierarchy_df = hierarchy_df[(hierarchy_df['level'] >= level_start) & (hierarchy_df['level'] <= level_end)].copy()
    missing_dfs = []
    for year in years:
        year_df = df[df['year_id'] == year].copy()
        missing_location_rows = level_hierarchy_df[~level_hierarchy_df['location_id'].isin(year_df['location_id'])]
        if not missing_location_rows.empty:
            # Create a DataFrame with the missing locations and set the count_variable to 0
            missing_df = pd.DataFrame({
                'location_id': missing_location_rows['location_id'].values,
                'year_id': year,
                variable: 0,
                'level': missing_location_rows['level'].values
            })
            missing_dfs.append(missing_df)
    # Check if missing dfs is not empty
    if missing_dfs:
        missing_df = pd.concat(missing_dfs, ignore_index=True).drop(columns=['level'])
        df = pd.concat([df, missing_df], ignore_index=True)
    return df

######### Raking functions #########
def prep_df(df, hierarchy_df):
    '''
    Adds the 'level' column to the DataFrame based on the hierarchy DataFrame.
    Removes 'parent_id' column if it exists.
    '''
    if 'level' not in df.columns:
        df = df.merge(hierarchy_df[['location_id', 'level']], on='location_id', how='left').copy()
    if 'parent_id' in df.columns:
        df = df.drop(columns=['parent_id'])
    return df

def rake_level(count_variable, level_df, level_m1_df, hierarchy_df):
    '''
    Rakes the level DataFrame to the next level using the hierarchy DataFrame.
    '''
    # Change the name of the count variable and prep for matching by parent_id from level
    level_m1_df = level_m1_df.rename(columns={
        count_variable: f'parent_{count_variable}',
        'location_id': 'parent_id'})
    # Prep the level df
    # Add in parent_id
    level_df = level_df.merge(
        hierarchy_df[['location_id', 'parent_id']],
        on='location_id',
        how='left'
    )
    # Aggregate the count variable by parent_id
    level_m1_agg_df= level_df.groupby(['parent_id', 'year_id']).agg({
        count_variable: 'sum',
        'population': 'sum'
    }).reset_index()
    # Merge in the level - 1 df
    level_m1_agg_df = level_m1_agg_df.merge(
        level_m1_df[['year_id', 'parent_id', f'parent_{count_variable}']],
        on=['year_id', 'parent_id'],
        how='left'
    )
    # Calculate the raking factor
    level_m1_agg_df['use_population'] = False
    level_m1_agg_df['raking_factor'] = level_m1_agg_df[f'parent_{count_variable}'] / level_m1_agg_df[count_variable]
    # Set the raking factor to 1 where the parent count variable is 0
    level_m1_agg_df.loc[level_m1_agg_df[f'parent_{count_variable}'] == 0, 'raking_factor'] = 0
    # Set the
    # Print all the rows where level_m1_agg_df[f'agg_{count_variable}'] != 0 and level_m1_agg_df[count_variable] == 0
    mask = (level_m1_agg_df[f'parent_{count_variable}'] != 0) & (level_m1_agg_df[count_variable] == 0)
    if mask.any():
        mask_df = level_m1_agg_df[mask].copy()
        mask_df['raking_factor'] = level_m1_agg_df[f'parent_{count_variable}'] / level_m1_agg_df['population']
        mask_df['use_population'] = True
        # Replace the masked rows in level_m1_agg_df with the mask_df
        level_m1_agg_df.loc[mask, 'raking_factor'] = mask_df['raking_factor']
        level_m1_agg_df.loc[mask, 'use_population'] = mask_df['use_population']
    # Set the raking factor to 1 where the use
    level_df = level_df.merge(
        level_m1_agg_df[['year_id', 'parent_id', 'raking_factor', 'use_population']],
        on=['year_id', 'parent_id'],
        how='left'
    )

    # For rows where use_population is True
    population_mask = level_df['use_population'] == True
    count_mask = level_df['use_population'] == False
    level_df.loc[count_mask, count_variable] = level_df.loc[count_mask, count_variable] * level_df.loc[count_mask, 'raking_factor']
    level_df.loc[population_mask, count_variable] = level_df.loc[population_mask, 'population'] * level_df.loc[population_mask, 'raking_factor']
        # Apply the raking factor to the count variable
        
    
    # Drop the raking factor
    level_df = level_df.drop(columns=['raking_factor', 'parent_id', 'use_population'])
    #
    return level_df

def rake_aa_count_lsae_to_gbd(count_variable, hierarchy_df, aa_gbd_count_df, aa_lsae_count_df, aa_full_count_df_path, return_full_df=False):
    '''
    Rakes the LSAE age-aggregated data to match the GBD age-aggregated data.
    '''
    aa_gbd_count_df = prep_df(aa_gbd_count_df, hierarchy_df)
    aa_gbd_count_0_to_3_df = aa_gbd_count_df[aa_gbd_count_df['level'] <= 3].copy()
    aa_lsae_count_df = prep_df(aa_lsae_count_df, hierarchy_df)
    aa_lsae_count_df = make_aa_df_square(count_variable, aa_lsae_count_df, hierarchy_df, level_start = 3, level_end = 5)
    
    aa_gbd_count_df[f'{count_variable}_gbd'] = aa_gbd_count_df[count_variable]
    aa_lsae_count_df = aa_lsae_count_df.merge(
        aa_gbd_count_df[['location_id', 'year_id', f'{count_variable}_gbd']],
        on=['location_id', 'year_id'],
        how='left'
    )

    aa_lsae_count_df[count_variable] = aa_lsae_count_df[f'{count_variable}_gbd'].fillna(aa_lsae_count_df[count_variable])
    aa_lsae_count_df = aa_lsae_count_df.drop(columns=[f'{count_variable}_gbd'])

    # Rake 4 to 3 def make_aa_df_square(variable, df, hierarchy_df, level_start, level_end):
    level_m1_df = aa_gbd_count_0_to_3_df[aa_gbd_count_0_to_3_df['level'] == 3].copy()
    level_df = aa_lsae_count_df[aa_lsae_count_df['level'] == 4].copy()
    level_df = make_aa_df_square(count_variable, level_df, hierarchy_df, 4, 4)
    level_4_df = rake_level(count_variable, level_df, level_m1_df, hierarchy_df)
    # Rake 5 to 4
    level_m1_df = level_4_df.copy()
    level_df = aa_lsae_count_df[aa_lsae_count_df['level'] == 5].copy()
    level_df = make_aa_df_square(count_variable, level_df, hierarchy_df, 5, 5)
    level_5_df = rake_level(count_variable, level_df, level_m1_df, hierarchy_df)
    # Make aa_full_df
    aa_full_count_df = pd.concat([
        aa_gbd_count_0_to_3_df,
        level_4_df,
        level_5_df
    ], ignore_index=True)
    # Drop level column if it exists
    if 'level' in aa_full_count_df.columns:
        aa_full_count_df = aa_full_count_df.drop(columns=['level'])

    # Save the aa_full_df
    if aa_full_count_df_path is not None:
        write_parquet(aa_full_count_df, aa_full_count_df_path)
    # Return the full DataFrame if requested, return nothing otherwise
    if return_full_df:
        return aa_full_count_df
    
def make_aa_rate_variable(count_variable, aa_full_count_df, aa_full_population_df, aa_full_rate_df_path, return_full_df=False):
    '''
    Makes the age-aggregated rate variable from the age-aggregated count variable and population data.
    '''
    # Get the rate variable name e.g., malaria_mort_count -> malaria_mort_rate
    rate_variable = count_variable.replace('count', 'rate')
    # Merge the full aa count df with the population df
    aa_full_rate_df = aa_full_count_df.merge(aa_full_population_df, on=['location_id', 'year_id'], how='left')
    # Calculate the rate
    aa_full_rate_df[rate_variable] = aa_full_rate_df[count_variable] / aa_full_rate_df['population']
    # Drop the population column and the count variable
    aa_full_rate_df = aa_full_rate_df.drop(columns=['population', count_variable])
    # Save the full aa rate df as a parquet file to the specified path
    if aa_full_rate_df_path is not None:
        write_parquet(aa_full_rate_df, aa_full_rate_df_path)
    # Return the full DataFrame if requested, return nothing otherwise
    if return_full_df:
        return aa_full_rate_df

######### Aggregation functions #########
def aggregate_level(count_variable, level_df, hierarchy_df):
    """
    Aggregate the count variable from the level_df to the next higher level in the hierarchy.
    
    Parameters:
    - count_variable: str, the name of the count variable to aggregate
    - level_df: DataFrame, the DataFrame for the current level
    - hierarchy_df: DataFrame, the hierarchy DataFrame
    
    Returns:
    - DataFrame with aggregated counts at the next lower level
    """
    level_df = level_df.merge(hierarchy_df[['location_id', 'parent_id']], on='location_id', how='left')
    aggregated_df = level_df.groupby(['parent_id', 'year_id'])[count_variable].sum().reset_index()
    aggregated_df = aggregated_df.rename(columns={'parent_id': 'location_id'})
    return aggregated_df

def aggregate_aa_count_lsae_to_gbd(count_variable, hierarchy_df, aa_lsae_count_df, aa_full_count_df_path=None, return_full_df=False):
    
    aa_lsae_count_df = prep_df(aa_lsae_count_df, hierarchy_df)
    level_5_df = aa_lsae_count_df[aa_lsae_count_df['level'] == 5].copy()
    all_years = aa_lsae_count_df['year_id'].unique()
    level_5_hierarchy_df = hierarchy_df[hierarchy_df['level'] == 5].copy()

    level_5_hierarchy_df = make_aa_df_square(count_variable, level_5_df, level_5_hierarchy_df, 5, 5)

    level_dfs = []
    level_dfs.append(level_5_df)
    level_df = level_5_df.copy()
    for level in range(4, -1, -1):
        level_p1_df = aggregate_level(count_variable, level_df, hierarchy_df)
        level_dfs.append(level_p1_df)
        level_df = level_p1_df.copy()
    
    aa_full_count_df = pd.concat(level_dfs, ignore_index=True)
    if aa_full_count_df_path is not None:
        write_parquet(aa_full_count_df, aa_full_count_df_path)
    # Return the full DataFrame if requested, return nothing otherwise
    if return_full_df:
        return aa_full_count_df

def make_aa_full_rate_df_from_aa_count_df(rate_variable, count_variable, aa_full_count_df, aa_full_population_df, aa_full_rate_df_path=None, return_full_df=False):
    if 'population' in aa_full_count_df.columns:
        aa_full_count_df = aa_full_count_df.drop(columns=['population'])

    aa_full_rate_df = aa_full_count_df.merge(aa_full_population_df[['location_id', 'year_id', 'population']], on=['location_id', 'year_id'], how='left').copy()
    aa_full_rate_df[rate_variable] = aa_full_rate_df[count_variable] / aa_full_rate_df['population']
    # Set rate to 0 where population is 0 to avoid division by zero
    aa_full_rate_df.loc[aa_full_rate_df['population'] == 0, rate_variable] = 0
    aa_full_rate_df = aa_full_rate_df.drop(columns=[count_variable, 'population'])
    if 'level' in aa_full_rate_df.columns:
        aa_full_rate_df = aa_full_rate_df.drop(columns=['level'])
    if aa_full_rate_df_path is not None:
        write_parquet(aa_full_rate_df, aa_full_rate_df_path)
    if return_full_df:
        return aa_full_rate_df


def aggregate_aa_rate_lsae_to_gbd(rate_variable, hierarchy_df, aa_lsae_rate_df, aa_full_population_df, aa_full_rate_df_path=None, return_full_df=False):
    if 'population' in aa_lsae_rate_df.columns:
        aa_lsae_rate_df = aa_lsae_rate_df.drop(columns=['population'])

    tmp_count_variable = 'tmp_count'
    aa_lsae_rate_df = prep_df(aa_lsae_rate_df, hierarchy_df)
    tmp_aa_lsae_count_df = aa_lsae_rate_df[aa_lsae_rate_df['level'] == 5].copy()
    tmp_aa_lsae_count_df = make_aa_df_square(tmp_count_variable, tmp_aa_lsae_count_df, hierarchy_df, level_start=5, level_end=5)
    tmp_aa_lsae_count_df = tmp_aa_lsae_count_df.merge(aa_full_population_df[['location_id', 'year_id', 'population']], on=['location_id', 'year_id'], how='left')
    tmp_aa_lsae_count_df[tmp_count_variable] = tmp_aa_lsae_count_df[rate_variable] * tmp_aa_lsae_count_df['population']
    tmp_aa_lsae_count_df = tmp_aa_lsae_count_df.drop(columns=[rate_variable, 'population'])
    aa_full_count_df = aggregate_aa_count_lsae_to_gbd(tmp_count_variable, hierarchy_df, tmp_aa_lsae_count_df, None, return_full_df=True)
    aa_full_rate_df = make_aa_full_rate_df_from_aa_count_df(rate_variable, tmp_count_variable, aa_full_count_df, 
                                                            aa_full_population_df, aa_full_rate_df_path = aa_full_rate_df_path, return_full_df = True)
    if 'level' in aa_full_rate_df.columns:
        aa_full_rate_df = aa_full_rate_df.drop(columns='level')
    if return_full_df:
        return aa_full_rate_df