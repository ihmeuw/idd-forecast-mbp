"""Shared utilities for stage-01 pixel-aggregation scripts.

Provides:
- load_raking_shapes: load admin polygons for a hierarchy (optionally clipped)
- blocks_with_shapefile_intersections: identify modeling-frame blocks whose
  footprint intersects ≥1 admin polygon in a hierarchy's raking shapefile

The block-intersection helper drives the "skip empty blocks" optimization in
the four stage-01 workflows. For lsae_1285, ~32% of the 784 modeling-frame
blocks are open ocean / Antarctic interior / otherwise uncovered tiles whose
pixel-aggregation outputs would be zero-row parquets. Filtering them out at
launcher time avoids spawning Slurm tasks that do real work and write empty
files; the hierarchy aggregator's sum-then-divide math is mathematically
unchanged by the skip.

Intersection is computed on the fly from the team-shared modeling frame +
raking shapefile (no persisted skip-list artifact). The inputs are stable;
the helper takes ~10-30 s per hierarchy.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore

from idd_forecast_mbp import constants as mbpc


RAKING_ROOT = Path("/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking")


def load_raking_shapes(
    full_aggregation_hierarchy: str,
    bounds: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Load shapes for a full aggregation hierarchy, optionally clipped to bounds.

    Parameters
    ----------
    full_aggregation_hierarchy
        The full aggregation hierarchy to load (e.g. "gbd_2021", "lsae_1285").
    bounds
        Optional (xmin, ymin, xmax, ymax) to subset the read at file-load time.
        If None, reads the full shapefile — needed by callers like
        blocks_with_shapefile_intersections that operate on the global extent.

    Returns
    -------
    gpd.GeoDataFrame
        The shapes for the given hierarchy and bounds. For gbd_* hierarchies
        the result is filtered to the most-detailed populated locations; for
        lsae_* hierarchies the file already contains only admin-2 polygons.
    """
    if full_aggregation_hierarchy in ["gbd_2021", "gbd_2023"]:
        shape_path = RAKING_ROOT / f"shapes_{full_aggregation_hierarchy}.parquet"
        gdf = gpd.read_parquet(shape_path, bbox=bounds)

        # Population file carries supplemented locations not in the modeled
        # GBD hierarchy (zero-pop or WPP-scalar places).
        pop_path = RAKING_ROOT / f"population_{full_aggregation_hierarchy}.parquet"
        pop = pd.read_parquet(pop_path)

        keep_cols = ["location_id", "location_name", "most_detailed", "parent_id"]
        keep_mask = (
            (pop.year_id == pop.year_id.max())
            & (pop.most_detailed == 1)
        )
        out = gdf.merge(pop.loc[keep_mask, keep_cols], on="location_id", how="left")
    elif full_aggregation_hierarchy in ["lsae_1209", "lsae_1285"]:
        shape_path = (
            RAKING_ROOT
            / "gbd-inputs"
            / f"shapes_{full_aggregation_hierarchy}_a2.parquet"
        )
        out = gpd.read_parquet(shape_path, bbox=bounds)
    else:
        raise ValueError(f"Unknown pixel hierarchy: {full_aggregation_hierarchy}")
    return out


def blocks_with_shapefile_intersections(
    hierarchy: str,
    modeling_frame_path: Path | str | None = None,
) -> set[str]:
    """Return the set of block_keys whose footprint intersects ≥1 admin polygon.

    For a given hierarchy's raking shapefile, identifies which modeling-frame
    blocks contain land covered by that hierarchy. Excluded blocks would
    contribute zero rows to the per-location accumulator in the hierarchy
    step's sum-then-divide rollup, so filtering them out is mathematically
    lossless.

    Intersection is purely geometric and time-invariant — the shapefile
    doesn't change with year. Safe to call once per launcher run.

    Parameters
    ----------
    hierarchy
        Hierarchy name passed through to load_raking_shapes
        (e.g. "lsae_1285", "gbd_2021").
    modeling_frame_path
        Optional override for the population-model modeling_frame.parquet
        location. Defaults to mbpc.MODELING_FRAME_PATH.

    Returns
    -------
    set[str]
        block_keys with at least one polygon intersection.
    """
    if modeling_frame_path is None:
        modeling_frame_path = mbpc.MODELING_FRAME_PATH
    modeling_frame = gpd.read_parquet(modeling_frame_path)
    blocks_gdf = (
        modeling_frame[["block_key", "geometry"]]
        .dissolve(by="block_key")
        .reset_index()
    )
    shapes = load_raking_shapes(hierarchy)  # bounds=None → full extent
    if shapes.crs != blocks_gdf.crs:
        shapes = shapes.to_crs(blocks_gdf.crs)
    joined = gpd.sjoin(blocks_gdf, shapes, how="inner", predicate="intersects")
    return set(joined["block_key"].unique())
