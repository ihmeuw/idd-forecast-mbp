import pandas as pd # type: ignore
from rra_tools import jobmon # type: ignore
from pathlib import Path # type: ignore
import geopandas as gpd # type: ignore
from rra_tools.shell_tools import mkdir # type: ignore
import numpy as np # type: ignore
import argparse
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.yaml import parse_yaml_dictionary
import yaml

# Sibling module — importable because Python adds the script's dir to sys.path[0]
from block_utils import blocks_with_shapefile_intersections  # noqa: E402

parser = argparse.ArgumentParser(description="Run James code")

# Define arguments
parser.add_argument("--covariate", type=str, required=True, help="covariate")
parser.add_argument("--hierarchy", type=str, required=True, help="Hierarchy")

# Parse arguments
args = parser.parse_args()

covariate = args.covariate
hierarchy = args.hierarchy

covariate_dict = parse_yaml_dictionary(covariate)
covariate_name = covariate_dict['covariate_name']
summary_statistic = covariate_dict['summary_statistic']
years = covariate_dict['years']
synoptic = covariate_dict['synoptic']
cc_sensitive = covariate_dict['cc_sensitive']

modeling_frame_path = "/mnt/team/rapidresponse/pub/population-model/modeling/100m/modeling_frame.parquet"

DATA_PATH = mbpc.MODEL_ROOT / "02-processed_data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

summary_covariate = f"{covariate_name}_{summary_statistic}"

HIERARCHY_MAP = {
    "gbd_2021": [
        "gbd_2021",
        "fhs_2021",
    ],  # GBD pixel hierarchy maps to GBD and FHS locations
    "lsae_1209": ["lsae_1209"],  # LSAE pixel hierarchy maps to LSAE locations
    "gbd_2023": ["gbd_2023"],  # :shrug: about fhs here
    "lsae_1285": ["lsae_1285"],  # LSAE pixel hierarchy maps to LSAE locations
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

        print("results.index[:5]:", results.index[:5])
        print("parent_map.index[:5]:", parent_map.index[:5])
        print("level:", level)

        # Continue aggregation only on the present locations
        subset = results.loc[parent_map.index]
        subset["parent_id"] = parent_map

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
    results["value"] = results["weighted_climate"] / results["population"].replace(0, np.nan)
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
    allowed_hierarchies = ["gbd_2021","gbd_2023", "fhs_2021", "lsae_1209", "lsae_1285"]
    if subset_hierarchy not in allowed_hierarchies:
        msg = f"Unknown admin hierarchy: {subset_hierarchy}"
        raise ValueError(msg)
    path = root / "gbd-inputs" / f"hierarchy_{subset_hierarchy}.parquet"
    return pd.read_parquet(path)

def post_process(df: pd.DataFrame, pop_df: pd.DataFrame) -> pd.DataFrame: # Fix this for other summary_variable/variable/etc
    """
    Rename 000 to {summary_covariate}_per_capita
    Merge in population
    Create {summary_covariate}_capita*population -> {summary_covariate}
    """

    # Rename 000 to people_flood_days_per_capita
    df = df.rename(columns={"000": f"{summary_covariate}_per_capita"})

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

    # Create {summary_covariate}
    full_df[summary_covariate] = (
        full_df[f"{summary_covariate}_per_capita"] * full_df["population"]
    ).astype(np.float32)

    return full_df


def hierarchy_main(
    covariate_name: str,
    summary_covariate: str,
    cc_sensitive: bool,
    hierarchy: str,
) -> None:
    # Load hierarchy data for aggregation
    hierarchy_df = pd.read_parquet(f"/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking/gbd-inputs/hierarchy_{hierarchy}.parquet")

    # Get all block keys, then drop those whose footprint doesn't intersect any
    # admin polygon in the hierarchy's raking shapefile. Skipped blocks would
    # have contributed zero rows under sum-then-divide aggregation, so this is
    # lossless. Must match the filter the launcher (02_pixel_main_parallel.py)
    # applied — otherwise we try to read parquets the launcher didn't write.
    modeling_frame = gpd.read_parquet(modeling_frame_path)
    block_keys = modeling_frame.block_key.unique()
    intersecting = blocks_with_shapefile_intersections(hierarchy)
    block_keys = [b for b in block_keys if b in intersecting]
    print(f"  Block keys after shapefile-intersection filter: {len(block_keys)}")

    DRAWS = [f"{d:>03}" for d in range(1)]

    if cc_sensitive:
        scenarios = ["ssp126", "ssp245", "ssp585"]
    else:
        scenarios = ["cc_insensitive"]

    for scenario in scenarios:
        all_results = []
        pop_df: pd.DataFrame | None = None

        for draw in DRAWS:
            draw_results = []
            for block_key in block_keys:
                draw_df = pd.read_parquet(DATA_PATH / "GBD2023" / hierarchy / covariate_name / block_key / f"{draw}.parquet")
                # filter by scenario
                draw_df = draw_df[draw_df["scenario"] == scenario]
                # drop scenario column
                draw_df = draw_df.drop(columns=["scenario"])

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
            subset_results["scenario"] = scenario

            # post-process the results
            subset_results = post_process(
                subset_results,
                pop_df,
            )

            # Save results for the subset hierarchy
            subset_results_path = DATA_PATH / subset_hierarchy
            filename = f"{summary_covariate}_{scenario}.parquet" 
            mkdir(subset_results_path, parents=True, exist_ok=True)
            subset_results.to_parquet(
                subset_results_path / filename,
                index=True,
            )
            final_path = subset_results_path / filename
            final_path.chmod(0o775)

            subset_pop = pop_df[pop_df["location_id"].isin(subset_location_ids)]
            popname = f"population.parquet"
            # Check if the population file already exists
            if (subset_results_path / popname).exists():
                # If it exists, don't re-write it
                continue
            else:
                # If it doesn't exist, write it
                subset_pop.to_parquet(
                    subset_results_path / popname,
                    index=True,
                )
                # change file permssions to 0775
                pop_path = subset_results_path / popname
                pop_path.chmod(0o775)


# Call the function with parsed arguments
hierarchy_main(
    covariate_name=covariate_name,
    summary_covariate=summary_covariate,
    cc_sensitive=cc_sensitive,
    hierarchy=hierarchy,
)