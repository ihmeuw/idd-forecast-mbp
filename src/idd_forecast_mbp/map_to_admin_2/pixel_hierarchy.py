import pandas as pd # type: ignore
from rra_tools import jobmon # type: ignore
from pathlib import Path # type: ignore
import geopandas as gpd # type: ignore
from rra_tools.shell_tools import mkdir # type: ignore
import numpy as np # type: ignore
import argparse
from rra_flooding.data import FloodingData
from rra_flooding import constants as rfc
from rra_flooding.helper_functions import parse_yaml_dictionary
import yaml

parser = argparse.ArgumentParser(description="Run James code")

# Define arguments
parser.add_argument("--variable", type=str, required=True, help="Variable to process")
parser.add_argument("--adjustment_num", type=int, required=True, help="Adjustment number")
parser.add_argument("--hierarchy", type=str, required=True, help="Hierarchy")
parser.add_argument("--scenario", type=str, required=True, help="Scenario")
parser.add_argument("--model", type=str, required=True, help="Model")
parser.add_argument("--variant", type=str, default="r1i1p1f1", help="Model variant identifier")


# Parse arguments
args = parser.parse_args()

variable = args.variable
adjustment_num = args.adjustment_num
hierarchy = args.hierarchy
scenario = args.scenario
model = args.model
variant = args.variant

variable_dict = parse_yaml_dictionary(variable, adjustment_num)
summary_variable = variable_dict['summary_variable']

HIERARCHY_MAP = {
    "gbd_2021": [
        "gbd_2021",
        "fhs_2021",
    ],  # GBD pixel hierarchy maps to GBD and FHS locations
    "lsae_1209": ["lsae_1209"],  # LSAE pixel hierarchy maps to LSAE locations
}

def aggregate_climate_to_hierarchy(
    data: pd.DataFrame, hierarchy: pd.DataFrame
) -> pd.DataFrame:
    """Create all aggregate climate values for a given hierarchy from most-detailed data.

    Parameters
    ----------
    data
        The most-detailed climate data to aggregate.
    hierarchy
        The hierarchy to aggregate the data to.

    Returns
    -------
    pd.DataFrame
        The climate data with values for all levels of the hierarchy.
    """
    results = data.set_index("location_id").copy()

    # Most detailed locations can be at multiple levels of the hierarchy,
    # so we loop over all levels from most detailed to global, aggregating
    # level by level and appending the results to the data.
    for level in reversed(list(range(1, hierarchy.level.max() + 1))):
        level_mask = hierarchy.level == level
        parent_map = hierarchy.loc[level_mask].set_index("location_id").parent_id

        # For every location in the parent map, we need to check if it is the results
        # For those that are, proceed to aggregate
        # For those that aren't, check to make sure their parent is in the results. If not, exit with an error
        absent_parent_map = parent_map.index.difference(results.index)
        if len(absent_parent_map) > 0:
            msg = f"Some parent locations are not in the results: {absent_parent_map}"
            # Check to see if the parent of each location id that is missing is in the results
            parent_of_absent = parent_map.loc[absent_parent_map]
            unique_parent_ids = parent_of_absent.unique()
            # Check to see if the unique_parent_ids are in the results
            missing_parents = unique_parent_ids[~np.isin(unique_parent_ids, results.index)]
            if len(missing_parents) > 0:
                msg = f"Some parent locations are not in the results: {missing_parents}"
                raise ValueError(msg)
        
        present_parent_map = parent_map.loc[parent_map.index.isin(results.index)]
        # Continue aggregation only on the present locations
        subset = results.loc[present_parent_map.index]
        subset["parent_id"] = present_parent_map

        parent_values = (
            subset.groupby(["year_id", "parent_id"])[["weighted_climate", "population"]]
            .sum()
            .reset_index()
            .rename(columns={"parent_id": "location_id"})
            .set_index("location_id")
        )
        results = pd.concat([results, parent_values])
    results = (
        results.reset_index()
        .sort_values(["location_id", "year_id"])
    )
    parent_values["value"] = parent_values.weighted_climate / parent_values.population
    return results

def load_subset_hierarchy(subset_hierarchy: str) -> pd.DataFrame:
    """Load a subset location hierarchy.

    The subset hierarchy might be equal to the full aggregation hierarchy,
    but it might also be a subset of the full aggregation hierarchy.
    These hierarchies are used to provide different views of aggregated
    climate data.

    Parameters
    ----------
    subset_hierarchy
        The administrative hierarchy to load (e.g. "gbd_2021")

    Returns
    -------
    pd.DataFrame
        The hierarchy data with parent-child relationships
    """
    root = Path("/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking")
    allowed_hierarchies = ["gbd_2021", "fhs_2021", "lsae_1209", "lsae_1285"]
    if subset_hierarchy not in allowed_hierarchies:
        msg = f"Unknown admin hierarchy: {subset_hierarchy}"
        raise ValueError(msg)
    path = root / "gbd-inputs" / f"hierarchy_{subset_hierarchy}.parquet"
    return pd.read_parquet(path)

def post_process(df: pd.DataFrame, pop_df: pd.DataFrame) -> pd.DataFrame: # Fix this for other summary_variable/variable/etc
    """
    Rename 000 to people_flood_days_per_capita
    Merge in population
    Create ppeople_flood_days_per_capita*population -> people_flood_days
    """

    # Rename 000 to people_flood_days_per_capita
    df = df.rename(columns={"000": "people_flood_days_per_capita"})

    # Merge in population
    full_df = df.merge(
        pop_df,
        on=["location_id", "year_id"],
        how="left",
    )
    # assert all location_ids and years combinations are present
    assert df.shape[0] == full_df.shape[0]
    assert df.location_id.nunique() == full_df.location_id.nunique()
    assert df.year_id.nunique() == full_df.year_id.nunique()

    # Create people_flood_days
    full_df["people_flood_days"] = (
        full_df["people_flood_days_per_capita"] * full_df["population"]
    ).astype(np.float32)

    return full_df


def hierarchy_main(
    hierarchy: str,
    scenario: str,
    model: str,
    variant: str,
) -> None:
    measure = summary_variable
    root = Path("/mnt/team/rapidresponse/pub/flooding/results/output/")

    # Load hierarchy data for aggregation
    hierarchy_df = pd.read_parquet(f"/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_{hierarchy}.parquet")

    # Get all block keys
    modeling_frame = gpd.read_parquet("/mnt/team/rapidresponse/pub/population-model/ihmepop_results/2025_03_22/modeling_frame.parquet")
    block_keys = modeling_frame.block_key.unique()

    DRAWS = [f"{d:>03}" for d in range(1)]


    all_results = []
    pop_df: pd.DataFrame | None = None

    for draw in DRAWS:
        draw_results = []
        for block_key in block_keys:
            draw_df = pd.read_parquet(root / "raw-results" / hierarchy / model / block_key / summary_variable / f"{draw}.parquet")
            # filter by scenario
            draw_df = draw_df[draw_df["scenario"] == scenario]
            # drop scenario and measure columns
            draw_df = draw_df.drop(columns=["scenario", "measure"])

            draw_results.append(draw_df)

        draw_df = (
            pd.concat(draw_results, ignore_index=True)
            .groupby(["location_id", "year_id"])
            .sum()
            .reset_index()
        )

        agg_df = aggregate_climate_to_hierarchy(
            draw_df,
            hierarchy_df,
        ).set_index(["location_id", "year_id"]).reset_index(drop=False)

        pop_df = agg_df[["location_id", "year_id", "population"]]
        pop_df = pop_df.set_index(["location_id", "year_id"]).reset_index(drop=False)

        agg_df["value"] = agg_df.weighted_climate / agg_df.population
        agg_df = agg_df[["location_id", "year_id", "value"]].rename(columns={"value": draw})
        all_results.append(agg_df)

    
    combined_results = pd.concat(all_results, axis=1)

    # Produce views for subset hierarchies
    subset_hierarchies = HIERARCHY_MAP[hierarchy]
    for subset_hierarchy in subset_hierarchies:
        # Load the subset hierarchy
        subset_hierarchy_df = load_subset_hierarchy(subset_hierarchy)

        # Filter results to only include locations in the subset hierarchy
        subset_location_ids = subset_hierarchy_df["location_id"].tolist()
        subset_results = combined_results[combined_results["location_id"].isin(subset_location_ids)]

        # add columns model, scenario, variant
        subset_results["model"] = model
        subset_results["scenario"] = scenario
        subset_results["variant"] = variant

        # post-process the results
        subset_results = post_process(
            subset_results,
            pop_df,
        )

        # Save results for the subset hierarchy
        subset_results_path = (
            root / subset_hierarchy 
        )
        filename = f"{summary_variable}_{scenario}_{model}_{variant}.parquet" 
        mkdir(subset_results_path, parents=True, exist_ok=True)
        subset_results.to_parquet(
            subset_results_path / filename,
            index=True,
        )
        final_path = subset_results_path / filename
        final_path.chmod(0o775)

        save_population = (
            measure == summary_variable and scenario == "ssp245" and draw == "000" and scenario == "ssp245" 
        )
        if save_population:
            subset_pop = pop_df[pop_df["location_id"].isin(subset_location_ids)]
            popname = f"population.parquet"
            subset_pop.to_parquet(
                subset_results_path / popname,
                index=True,
            )
            # change file permssions to 0775
            pop_path = subset_results_path / popname
            # pop_path.chmod(0o775)


# Call the function with parsed arguments
hierarchy_main(
    hierarchy=hierarchy,
    scenario=scenario,
    model=model,
    variant=variant,
)