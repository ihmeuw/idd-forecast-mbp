import pandas as pd
import numpy as np
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.xarray_functions import convert_to_xarray, write_netcdf
from idd_forecast_mbp.parquet_functions import read_parquet_with_integer_ids, write_parquet

RAW_DATA_PATH = rfc.MODEL_ROOT / "01-raw_data"
PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"

GBD_DATA_PATH = f"{RAW_DATA_PATH}/gbd"
lsae_hierarchy = "lsae_1209"

################################################################
#### Hierarchy Paths, loading, and cleaning
################################################################
lsae_2023_hierarchy_path = "/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_lsae_1209.parquet"
gbd_2023_hierarchy_path = f"{GBD_DATA_PATH}/gbd_2023_modeling_hierarchy.parquet"
fhs_2023_hierarchy_path = f"{GBD_DATA_PATH}/fhs_2023_modeling_hierarchy.parquet"

# Output path for the full hierarchy
hierarchy_2023_df_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_{lsae_hierarchy}.parquet"
hierarchy_2023_ds_path = f"{PROCESSED_DATA_PATH}/full_hierarchy_2023_{lsae_hierarchy}.nc"

lsae_2023_hierarchy_df = read_parquet_with_integer_ids(lsae_2023_hierarchy_path)
gbd_2023_hierarchy_df = read_parquet_with_integer_ids(gbd_2023_hierarchy_path)
fhs_2023_hierarchy_df = read_parquet_with_integer_ids(fhs_2023_hierarchy_path)

columns_to_drop_all = [
    'local_id',
    "location_ascii_name",
    "location_name_short",
    "location_name_medium",
    "location_type_id",
    "developed"
]
columns_to_drop_some = [
    "start_date",
    "end_date",
    "date_inserted",
    "last_updated",
    "last_updated_by",
    "last_updated_action"
]

gbd_2023_hierarchy_df = gbd_2023_hierarchy_df.drop(columns=columns_to_drop_all +  ["lancet_label","who_label"])
fhs_2023_hierarchy_df = fhs_2023_hierarchy_df.drop(columns=columns_to_drop_all +  ["lancet_label","who_label"])
lsae_2023_hierarchy_df = lsae_2023_hierarchy_df.drop(columns=columns_to_drop_all + columns_to_drop_some)

india_subnats = gbd_2023_hierarchy_df[gbd_2023_hierarchy_df['parent_id'] == 163]['location_id'].unique().tolist()
india_urban_rural = gbd_2023_hierarchy_df[gbd_2023_hierarchy_df['parent_id'].isin(india_subnats)]['location_id'].unique().tolist()
# Remove India urban/rural locations from the GBD hierarchy
gbd_2023_hierarchy_df = gbd_2023_hierarchy_df[~gbd_2023_hierarchy_df['location_id'].isin(india_urban_rural)]

################################################################
#### Make Full 2023 Hierarchy
################################################################



hierarchy_2023_df = gbd_2023_hierarchy_df.copy()
hierarchy_2023_df['location_set_version_id'] = lsae_2023_hierarchy_df['location_set_version_id'][0]
hierarchy_2023_df['location_set_id'] = lsae_2023_hierarchy_df['location_set_id'][0]
# Rename most_detailed to most_detailed_gbd
hierarchy_2023_df = hierarchy_2023_df.rename(columns={"most_detailed": "most_detailed_gbd"})
hierarchy_2023_df['most_detailed_lsae'] = 0
lsae_2023_hierarchy_df = lsae_2023_hierarchy_df.rename(columns={"most_detailed": "most_detailed_lsae"})
lsae_2023_hierarchy_df['most_detailed_gbd'] = 0
#
lsae_level_1 = lsae_2023_hierarchy_df[lsae_2023_hierarchy_df['level'] == 1]

for _, row in lsae_level_1.iterrows():
    # Get location_id from the Series
    location_id = row.location_id

    # Find all children of this location
    children_rows = lsae_2023_hierarchy_df[lsae_2023_hierarchy_df['parent_id'] == location_id]

    # Only proceed if this location exists in the full hierarchy
    if location_id in hierarchy_2023_df['location_id'].values:
        location_row = hierarchy_2023_df[hierarchy_2023_df['location_id'] == location_id]
        parent_path = location_row['path_to_top_parent'].values[0]
        parent_sort_order = location_row['sort_order'].values[0]

        # Prepare children rows
        new_children_rows = children_rows.copy()
        new_children_rows['path_to_top_parent'] = new_children_rows['location_id'].apply(lambda x: f"{parent_path},{x}")
        new_children_rows['level'] = 4
        new_children_rows['super_region_id'] = location_row['super_region_id'].values[0]
        new_children_rows['super_region_name'] = location_row['super_region_name'].values[0]
        new_children_rows['region_id'] = location_row['region_id'].values[0]
        new_children_rows['region_name'] = location_row['region_name'].values[0]

        # Prepare grandchildren rows
        all_new_rows = [new_children_rows]
        for _, child_row in new_children_rows.iterrows():
            child_location_id = child_row['location_id']
            grandchildren_rows = lsae_2023_hierarchy_df[lsae_2023_hierarchy_df['parent_id'] == child_location_id].copy()
            child_path = child_row['path_to_top_parent']
            grandchildren_rows['path_to_top_parent'] = grandchildren_rows['location_id'].apply(lambda x: f"{child_path},{x}")
            grandchildren_rows['level'] = 5
            grandchildren_rows['super_region_id'] = location_row['super_region_id'].values[0]
            grandchildren_rows['super_region_name'] = location_row['super_region_name'].values[0]
            grandchildren_rows['region_id'] = location_row['region_id'].values[0]
            grandchildren_rows['region_name'] = location_row['region_name'].values[0]
            all_new_rows.append(grandchildren_rows)

        # Concatenate all new rows
        new_rows = pd.concat(all_new_rows, ignore_index=True)

        # Sort and assign sort_order
        new_rows = new_rows.sort_values(by=['sort_order'])
        num_rows = new_rows.shape[0]
        new_rows['sort_order'] = [parent_sort_order + (i + 1) / (num_rows + 2) for i in range(num_rows)]

        # Add to the full hierarchy
        hierarchy_2023_df = pd.concat([hierarchy_2023_df, new_rows], ignore_index=True)

# Sort the hierarchy_2023_df by sort_order
hierarchy_2023_df = hierarchy_2023_df.sort_values(by=['sort_order'])
# Replace sort_order with a sequence of numbers that start with 1 and increment by 1
hierarchy_2023_df['sort_order'] = [i + 1 for i in range(hierarchy_2023_df.shape[0])]

# Check for duplicates
duplicate_locs = hierarchy_2023_df.groupby('location_id').size()
duplicate_locs = duplicate_locs[duplicate_locs > 1]

if len(duplicate_locs) > 0:
    for loc_id in duplicate_locs.index:
        # Get rows for this location
        loc_rows = hierarchy_2023_df[hierarchy_2023_df['location_id'] == loc_id]
        
        if len(loc_rows) > 2:
           print(f"Location {loc_id} has more than 2 rows")
            
        # Count rows where most_detailed_gbd = 0
        gbd_false = (loc_rows['most_detailed_gbd'] == 0).sum()
        gbd_true = (loc_rows['most_detailed_gbd'] == 1).sum()
        
        if gbd_true == 0:
            # Delete all but the first row (keep only one non-detailed row)
            rows_to_keep = loc_rows.index[0:1]  # Keep first row only
            rows_to_remove = loc_rows.index[1:]  # Remove the rest
            hierarchy_2023_df = hierarchy_2023_df.drop(rows_to_remove)
        
        else:
            # Keep the detailed row, remove non-detailed rows
            rows_to_remove = loc_rows[loc_rows['most_detailed_gbd'] == 0].index
            hierarchy_2023_df = hierarchy_2023_df.drop(rows_to_remove)


################################################################
#### Add columns documenting who was in which OG hierarchy
################################################################
# Add a column to indicate if the location is in the FHS hierarchy
hierarchy_2023_df['in_fhs_hierarchy'] = hierarchy_2023_df['location_id'].isin(fhs_2023_hierarchy_df['location_id'])
# Add a column to indicate if the location is in the LSAE hierarchy
hierarchy_2023_df['in_lsae_hierarchy'] = hierarchy_2023_df['location_id'].isin(lsae_2023_hierarchy_df['location_id'])
# Add a column to indicate if the location is in the GBD hierarchy
hierarchy_2023_df['in_gbd_hierarchy'] = hierarchy_2023_df['location_id'].isin(gbd_2023_hierarchy_df['location_id'])
# Add a column to indicate if the locaiton is a "most_detailed" location in the GBD hierarchy
hierarchy_2023_df['most_detailed_fhs'] = hierarchy_2023_df['location_id'].isin(fhs_2023_hierarchy_df[fhs_2023_hierarchy_df['most_detailed'] == 1]['location_id'])
# Check to see if any location is in the fhs hierarchy but not in hierarchy_2023_df
if not fhs_2023_hierarchy_df['location_id'].isin(hierarchy_2023_df['location_id']).all():
    raise ValueError("Some locations in the FHS hierarchy are not in the full hierarchy")
# Check to see if any location is in the lsae hierarchy and in the GBD hierarchy but not in hierarchy_2023_df
missing_lsae_locs = lsae_2023_hierarchy_df[~lsae_2023_hierarchy_df['location_id'].isin(hierarchy_2023_df['location_id'])]
# Are any of the missing locations in the GBD hierarchy?
missing_lsae_locs_in_gbd = missing_lsae_locs[missing_lsae_locs['location_id'].isin(gbd_2023_hierarchy_df['location_id'])]
if not missing_lsae_locs_in_gbd.empty:
    raise ValueError("Some locations in the LSAE hierarchy are in the GBD hierarchy but not in the full hierarchy")
len(missing_lsae_locs)

################################################################
#### Lookup table creation for LSAE to FHS hierarchy mapping
################################################################

fhs_hierarchy_location_ids = fhs_2023_hierarchy_df["location_id"].unique().tolist()

fhs_look_uptable_df = hierarchy_2023_df[["location_id", "location_name", "level", "parent_id", "most_detailed_lsae", "most_detailed_fhs"]].copy()
fhs_look_uptable_df["fhs_location_id"] = None
fhs_look_uptable_df["fhs_level"] = None
# Create lookup dictionaries for faster access
fhs_hierarchy_dict = dict(zip(fhs_2023_hierarchy_df["location_id"], fhs_2023_hierarchy_df["level"]))
hierarchy_parent_dict = dict(zip(hierarchy_2023_df["location_id"], hierarchy_2023_df["parent_id"]))

# Prepare result containers
total_rows = len(fhs_look_uptable_df)
most_detailed_fhs = [None] * total_rows
fhs_location_ids = [None] * total_rows
fhs_levels = [None] * total_rows

print(f"Processing {total_rows} locations...")

# Process in batches and print less frequently
for i, row in enumerate(fhs_look_uptable_df.itertuples()):
    location_id = row.location_id
    parent_id = row.parent_id
    level = row.level
    
    # Direct match
    if location_id in fhs_hierarchy_location_ids:
        most_detailed_fhs[i] = 1
        fhs_location_ids[i] = location_id
        fhs_levels[i] = level
    # Parent match
    elif parent_id in fhs_hierarchy_location_ids:
        most_detailed_fhs[i] = 0
        fhs_location_ids[i] = parent_id
        fhs_levels[i] = fhs_hierarchy_dict[parent_id]
    # Ancestor match
    elif parent_id in hierarchy_parent_dict:
        ancestor_id = hierarchy_parent_dict[parent_id]
        if ancestor_id in fhs_hierarchy_location_ids:
            most_detailed_fhs[i] = 0
            fhs_location_ids[i] = ancestor_id
            fhs_levels[i] = fhs_hierarchy_dict[ancestor_id]

# Update the dataframe all at once
fhs_look_uptable_df["most_detailed_fhs"] = most_detailed_fhs
fhs_look_uptable_df["fhs_location_id"] = fhs_location_ids
fhs_look_uptable_df["fhs_level"] = fhs_levels

# Write the updated DataFrame to a new parquet file
output_path = f"{PROCESSED_DATA_PATH}/lsae_to_fhs_table.parquet"
fhs_look_uptable_df.to_parquet(output_path, index=False)

# Merge the lookup table with the full hierarchy
hierarchy_2023_df = hierarchy_2023_df.merge(
    fhs_look_uptable_df[["location_id", "fhs_location_id", "fhs_level"]],
    on="location_id",
    how="left"
)

################################################################
#### Lookup table creation for LSAE to GBD hierarchy mapping
################################################################

gbd_hierarchy_location_ids = gbd_2023_hierarchy_df["location_id"].unique().tolist()

gbd_look_uptable_df = hierarchy_2023_df[["location_id", "location_name", "level", "parent_id", "most_detailed_lsae", "most_detailed_gbd"]].copy()
gbd_look_uptable_df["gbd_location_id"] = None
gbd_look_uptable_df["gbd_level"] = None
# Create lookup dictionaries for faster access
gbd_hierarchy_dict = dict(zip(gbd_2023_hierarchy_df["location_id"], gbd_2023_hierarchy_df["level"]))
hierarchy_parent_dict = dict(zip(hierarchy_2023_df["location_id"], hierarchy_2023_df["parent_id"]))

# Prepare result containers
total_rows = len(gbd_look_uptable_df)
most_detailed_gbd = [None] * total_rows
gbd_location_ids = [None] * total_rows
gbd_levels = [None] * total_rows

print(f"Processing {total_rows} locations...")

# Process in batches and print less frequently
for i, row in enumerate(gbd_look_uptable_df.itertuples()):
    location_id = row.location_id
    parent_id = row.parent_id
    level = row.level
    
    # Direct match
    if location_id in gbd_hierarchy_location_ids:
        most_detailed_gbd[i] = 1
        gbd_location_ids[i] = location_id
        gbd_levels[i] = level
    # Parent match
    elif parent_id in gbd_hierarchy_location_ids:
        most_detailed_gbd[i] = 0
        gbd_location_ids[i] = parent_id
        gbd_levels[i] = gbd_hierarchy_dict[parent_id]
    # Ancestor match
    elif parent_id in hierarchy_parent_dict:
        ancestor_id = hierarchy_parent_dict[parent_id]
        if ancestor_id in gbd_hierarchy_location_ids:
            most_detailed_gbd[i] = 0
            gbd_location_ids[i] = ancestor_id
            gbd_levels[i] = gbd_hierarchy_dict[ancestor_id]

# Update the dataframe all at once
gbd_look_uptable_df["most_detailed_gbd"] = most_detailed_gbd
gbd_look_uptable_df["gbd_location_id"] = gbd_location_ids
gbd_look_uptable_df["gbd_level"] = gbd_levels

# Write the updated DataFrame to a new parquet file
output_path = f"{PROCESSED_DATA_PATH}/lsae_to_gbd_table.parquet"
gbd_look_uptable_df.to_parquet(output_path, index=False)

# Merge the lookup table with the full hierarchy
hierarchy_2023_df = hierarchy_2023_df.merge(
    gbd_look_uptable_df[["location_id", "gbd_location_id", "gbd_level"]],
    on="location_id",
    how="left"
)

################################################################
#### Add a column for A0 locations
################################################################

A0_hierarchy_df = hierarchy_2023_df[["location_id", "parent_id", "level", "location_name", "path_to_top_parent"]].copy()
# Subset to level 3 locations and call it A0_hierarchy_df
A0_hierarchy_df = A0_hierarchy_df[A0_hierarchy_df["level"] == 3]
# Find all the unique location_ids in the A0_hierarchy_df
A0_location_ids = A0_hierarchy_df["location_id"].unique()
# Create a new column in hierarchy_2023_df called A0_location_id
hierarchy_2023_df["A0_location_id"] = np.nan
# For every row of hierarchy_2023_df, look at the numbers in the path_to_top_parent, find which one of them is an element of A0_location_ids and place that in the A0_location_id column
# Note: and example path_to_top_parent is 1,311,61467,69489
# In this case, 311 is an element of the A0_location_ids
# So the A0_location_id column will be 311
for i, row in hierarchy_2023_df.iterrows():
    # Split the path_to_top_parent by ","
    path = row["path_to_top_parent"].split(",")
    # Find the first element of path that is in A0_location_ids
    for loc in path:
        if int(loc) in A0_location_ids:
            hierarchy_2023_df.at[i, "A0_location_id"] = int(loc)
            break

################################################################
#### Save
################################################################
# Save the updated DataFrame to a new parquet file
write_parquet(hierarchy_2023_df, hierarchy_2023_df_path)

################################################################
#### Convert to xarray and save as NetCDF
################################################################
# Manually ensure compatible dtypes
hierarchy_2023_df['location_id'] = hierarchy_2023_df['location_id'].astype('int32')
# Convert to xarray 
hierarchy_2023_ds = convert_to_xarray(
    hierarchy_2023_df,
    dimensions=['location_id'],
    dimension_dtypes={'location_id': 'int32'},
    auto_optimize_dtypes=True
)

# Write to NetCDF
write_netcdf(hierarchy_2023_ds, hierarchy_2023_ds_path)