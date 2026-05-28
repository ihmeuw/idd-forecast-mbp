"""
Xarray operations for formatting data for FHS upload.
"""

from __future__ import annotations

import xarray as xr


def aggregate_locations(
    count_ds: xr.Dataset,
    pop_ds: xr.Dataset,
    source_locs: list[int],
    target_loc: int,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Sum a set of locations into a single target location, in count space.

    Used to aggregate Ethiopian sub-nationals (60908, 95069, 94364) into the
    national location (44858) required by the FHS upload format.

    Locations in source_locs that are not present in count_ds are silently
    ignored. If none are present, the datasets are returned unchanged.

    Parameters
    ----------
    count_ds:
        Dataset of counts with a location_id dimension.
    pop_ds:
        Dataset of population with a location_id dimension.
    source_locs:
        Location IDs to aggregate together and remove.
    target_loc:
        Location ID to assign to the aggregated result.

    Returns
    -------
    Tuple of (count_ds, pop_ds) with source_locs replaced by target_loc.
    """
    present = [loc for loc in source_locs if loc in count_ds.location_id.values]
    if not present:
        return count_ds, pop_ds

    count_agg = count_ds.sel(location_id=present).sum(dim="location_id")
    pop_agg = pop_ds.sel(location_id=present).sum(dim="location_id")

    count_agg = count_agg.assign_coords(location_id=target_loc).expand_dims("location_id")
    pop_agg = pop_agg.assign_coords(location_id=target_loc).expand_dims("location_id")

    keep = [loc for loc in count_ds.location_id.values if loc not in present]
    count_ds = xr.concat([count_ds.sel(location_id=keep), count_agg], dim="location_id")
    pop_ds = xr.concat([pop_ds.sel(location_id=keep), pop_agg], dim="location_id")

    return count_ds, pop_ds


def counts_to_rates(
    count_da: xr.DataArray,
    pop_da: xr.DataArray,
) -> xr.DataArray:
    """Divide counts by population; return 0.0 where population is zero or missing.

    Parameters
    ----------
    count_da:
        DataArray of counts.
    pop_da:
        DataArray of population. Must be broadcastable against count_da.

    Returns
    -------
    DataArray of rates (count / population), with 0.0 where population <= 0.
    """
    return (count_da / pop_da.where(pop_da > 0)).fillna(0.0)
