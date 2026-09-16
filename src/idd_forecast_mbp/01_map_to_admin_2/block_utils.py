"""Shared utilities for stage-01 pixel-aggregation scripts.

Provides:
- load_raking_shapes: load admin polygons for a hierarchy (optionally clipped)
- blocks_with_shapefile_intersections: identify modeling-frame blocks whose
  footprint intersects ≥1 admin polygon in a hierarchy's raking shapefile
- NearestResampler: cached nearest-neighbor resampler from a fixed lat/lon
  source grid to a fixed equal-area destination raster. Build once per task,
  apply many times — replaces per-call rasterio.warp.reproject inside the
  pixel_main inner loop.

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
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
import rasterio.warp  # type: ignore
import rasterra as rt  # type: ignore
from affine import Affine  # type: ignore

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


class NearestResampler:
    """Cached nearest-neighbor resampler from a fixed lat/lon source grid
    to a fixed equal-area destination raster.

    Built once per task with the source coord arrays (lat, lon) and a
    destination raster template. The build step computes a source-pixel
    flat-index array (one int per destination cell) by reprojecting
    destination pixel centers into the source CRS and snapping to the
    nearest source pixel. Subsequent ``apply(src_2d)`` calls reduce to
    a ``numpy.ndarray.take`` plus an in-place NaN mask, replacing the
    per-call geometric reprojection that ``rasterio.warp.reproject``
    performs.

    Equivalence: bit-for-bit identical to
    ``to_raster(...).resample_to(target, "nearest").astype(np.float32)._ndarray``
    on the same input, verified by unit tests and multi-block A/B in
    the source repo (climate-data).

    Convention: source data is assumed to be in lat-descending order
    (typical for an xarray slice with ``slice(lat_max, lat_min)``).
    To match the existing pipeline's ``to_raster``, which flips
    ``data[::-1]`` so the underlying raster has row 0 = southernmost,
    the cached indices are computed against the *flipped* source
    orientation; :meth:`apply` performs the flip internally.
    """

    def __init__(
        self,
        flat_idx_safe: npt.NDArray[np.int64],
        in_bounds: npt.NDArray[np.bool_],
        src_h: int,
        src_w: int,
        dst_shape: tuple[int, int],
    ) -> None:
        self._flat_idx_safe = flat_idx_safe
        self._in_bounds = in_bounds
        self._src_h = src_h
        self._src_w = src_w
        self._dst_shape = dst_shape

    @classmethod
    def build(
        cls,
        src_lats: npt.NDArray[np.floating],
        src_lons: npt.NDArray[np.floating],
        src_crs: str,
        dst_template: rt.RasterArray,
    ) -> "NearestResampler":
        src_h, src_w = len(src_lats), len(src_lons)
        dlat = (src_lats[1:] - src_lats[:-1]).mean()
        dlon = (src_lons[1:] - src_lons[:-1]).mean()
        src_transform = Affine(
            a=dlon, b=0.0, c=src_lons[0],
            d=0.0, e=-dlat, f=src_lats[-1],
        )

        dst_transform = dst_template.transform
        dst_crs = dst_template.crs
        dst_h, dst_w = dst_template._ndarray.shape

        dst_cols, dst_rows = np.meshgrid(
            np.arange(dst_w) + 0.5,
            np.arange(dst_h) + 0.5,
        )
        dst_xs = dst_transform.a * dst_cols + dst_transform.b * dst_rows + dst_transform.c
        dst_ys = dst_transform.d * dst_cols + dst_transform.e * dst_rows + dst_transform.f

        src_xs, src_ys = rasterio.warp.transform(
            dst_crs, src_crs,
            dst_xs.ravel().tolist(),
            dst_ys.ravel().tolist(),
        )
        src_xs = np.array(src_xs).reshape(dst_h, dst_w)
        src_ys = np.array(src_ys).reshape(dst_h, dst_w)

        inv = ~src_transform
        src_pix_cols = inv.a * src_xs + inv.b * src_ys + inv.c
        src_pix_rows = inv.d * src_xs + inv.e * src_ys + inv.f

        row_idx = np.floor(src_pix_rows).astype(np.int32)
        col_idx = np.floor(src_pix_cols).astype(np.int32)

        in_bounds = (
            (row_idx >= 0) & (row_idx < src_h)
            & (col_idx >= 0) & (col_idx < src_w)
        )
        row_idx_clipped = np.clip(row_idx, 0, src_h - 1)
        col_idx_clipped = np.clip(col_idx, 0, src_w - 1)

        flat_idx = row_idx_clipped.astype(np.int64) * src_w + col_idx_clipped.astype(np.int64)
        flat_idx_safe = np.where(in_bounds, flat_idx, 0)

        return cls(
            flat_idx_safe=flat_idx_safe,
            in_bounds=in_bounds,
            src_h=src_h,
            src_w=src_w,
            dst_shape=(dst_h, dst_w),
        )

    @property
    def src_shape(self) -> tuple[int, int]:
        return (self._src_h, self._src_w)

    @property
    def dst_shape(self) -> tuple[int, int]:
        return self._dst_shape

    def apply(
        self,
        src_2d: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.float32]:
        if src_2d.shape != (self._src_h, self._src_w):
            msg = (
                f"src_2d shape {src_2d.shape} does not match expected "
                f"({self._src_h}, {self._src_w})"
            )
            raise ValueError(msg)
        src_flat = src_2d[::-1].astype(np.float32, copy=False).ravel()
        out = src_flat.take(self._flat_idx_safe).reshape(self._dst_shape)
        np.putmask(out, ~self._in_bounds, np.float32(np.nan))
        return out
