import xarray as xr # type: ignore
from pathlib import Path
import numpy as np # type: ignore
import geopandas as gpd # type: ignore
from rasterio.features import rasterize # type: ignore
import rasterra as rt # type: ignore
from affine import Affine # type: ignore
from typing import cast
import numpy.typing as npt # type: ignore
import pandas as pd # type: ignore
import shapely # type: ignore
from rasterio.features import MergeAlg, rasterize # type: ignore
from shapely import MultiPolygon, Polygon # type: ignore
from typing import Literal, NamedTuple
import itertools
from rra_tools.shell_tools import mkdir # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.yaml_functions import load_yaml_dictionary, parse_yaml_dictionary
import argparse
import yaml

parser = argparse.ArgumentParser(description="Run James code")

# Define arguments
parser.add_argument("--threshold", type=float, required=True, help="urban threshold")
parser.add_argument("--hiearchy", type=str, required=True, help="Hiearchy")
parser.add_argument("--block_key", type=str, required=True, help="Block Key")

# Parse arguments
args = parser.parse_args()

threshold = args.threshold
hiearchy = args.hiearchy
block_key = args.block_key

years = list(range(2000, 2101))

POPULATION_1km_PATH = "/mnt/team/rapidresponse/pub/population-model/results/2025_02_21/world_cylindrical_1000/{year}q1.tif"
pop_root = Path("/mnt/team/rapidresponse/pub/population-model/modeling/100m/models/2025_02_19.001/raked_predictions")

DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"

class _Scenarios(NamedTuple):
    historical: str = "historical"
    ssp126: str = "ssp126"
    ssp245: str = "ssp245"
    ssp585: str = "ssp585"

SCENARIOS = _Scenarios()

AGGREGATION_SCENARIOS = [
    SCENARIOS.ssp126,
    SCENARIOS.ssp245,
    SCENARIOS.ssp585,
]

MAX_BOUNDS = {
    "ESRI:54034": (-20037508.34, 20037508.34, -6363885.33, 6363885.33),
}

inflation_factor = 0.01

def get_bbox(raster: rt.RasterArray, crs: str | None = None) -> shapely.Polygon:
    """Get the bounding box of a raster array.

    Parameters
    ----------
    raster
        The raster array to get the bounding box of.
    crs
        The CRS to return the bounding box in. If None, the bounding box
        is returned in the CRS of the raster.

    Returns
    -------
    shapely.Polybon
        The bounding box of the raster in the CRS specified by the crs parameter.
    """
    if raster.crs not in MAX_BOUNDS:
        msg = f"Unsupported CRS: {raster.crs}"
        raise ValueError(msg)

    xmin_clip, xmax_clip, ymin_clip, ymax_clip = MAX_BOUNDS[raster.crs]
    xmin, xmax, ymin, ymax = raster.bounds

    xmin = np.clip(xmin, xmin_clip, xmax_clip)
    xmax = np.clip(xmax, xmin_clip, xmax_clip)
    ymin = np.clip(ymin, ymin_clip, ymax_clip)
    ymax = np.clip(ymax, ymin_clip, ymax_clip)

    bbox = gpd.GeoSeries([shapely.box(xmin, ymin, xmax, ymax)], crs=raster.crs)
    out_bbox = bbox.to_crs(crs) if crs is not None else bbox.copy()

    # Check that our transformation didn't do something weird
    # (e.g. artificially clip the bounds or have the bounds extend over the
    # antimeridian)
    check_bbox = out_bbox.to_crs(raster.crs)
    area_change = (np.abs(bbox.area - check_bbox.area) / bbox.area).iloc[0]
    tolerance = 1e-6
    if area_change > tolerance:
        msg = f"Area change: {area_change}"
        raise ValueError(msg)

    return cast(shapely.Polygon, out_bbox.iloc[0])


def load_raking_shapes(full_aggregation_hierarchy: str, bounds: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Load shapes for a full aggregation hierarchy within given bounds.

    Parameters
    ----------
    full_aggregation_hierarchy
        The full aggregation hierarchy to load (e.g. "gbd_2021")
    bounds
        The bounds to load (xmin, ymin, xmax, ymax)

    Returns
    -------
    gpd.GeoDataFrame
        The shapes for the given hierarchy and bounds
    """
    root = Path("/mnt/team/rapidresponse/pub/population-model/admin-inputs/raking")
    if full_aggregation_hierarchy in ["gbd_2021", "gbd_2023"]:
        shape_path = (
            root/ f"shapes_{full_aggregation_hierarchy}.parquet"
        )
        gdf = gpd.read_parquet(shape_path, bbox=bounds)

        # We're using population data here instead of a hierarchy because
        # The populations include extra locations we've supplemented that aren't
        # modeled in GBD (e.g. locations with zero population or places that
        # GBD uses population scalars from WPP to model)
        pop_path = (
            root / f"population_{full_aggregation_hierarchy}.parquet"
        )
        pop = pd.read_parquet(pop_path)

        keep_cols = ["location_id", "location_name", "most_detailed", "parent_id"]
        keep_mask = (
            (pop.year_id == pop.year_id.max())  # Year doesn't matter
            & (pop.most_detailed == 1)
        )
        out = gdf.merge(pop.loc[keep_mask, keep_cols], on="location_id", how="left")
    elif full_aggregation_hierarchy in ["lsae_1209", "lsae_1285"]:
        # This is only a2 geoms, so already most detailed
        shape_path = (
            root
            / "gbd-inputs"
            / f"shapes_{full_aggregation_hierarchy}_a2.parquet"
        )
        out = gpd.read_parquet(shape_path, bbox=bounds)
    else:
        msg = f"Unknown pixel hierarchy: {full_aggregation_hierarchy}"
        raise ValueError(msg)
    return out

def build_bounds_map(
    raster_template: rt.RasterArray,
    shape_values: list[tuple[Polygon | MultiPolygon, int]],
) -> dict[int, tuple[slice, slice]]:
    """Build a map of location IDs to buffered slices of the raster template.

    Parameters
    ----------
    raster_template
        The raster template to build the bounds map for.
    shape_values
        A list of tuples where the first element is a shapely Polygon or MultiPolygon
        in the CRS of the raster template and the second element is the location ID
        of the shape.

    Returns
    -------
    dict[int, tuple[slice, slice]]
        A dictionary mapping location IDs to a tuple of slices representing the bounds
        of the location in the raster template. The slices are buffered by 10 pixels
        to ensure that the entire shape is included in the mask.
    """
    # The tranform maps pixel coordinates to the CRS coordinates.
    # This mask is the inverse of that transform.
    to_pixel = ~raster_template.transform

    bounds_map = {}
    for shp, loc_id in shape_values:
        xmin, ymin, xmax, ymax = shp.bounds
        pxmin, pymin = to_pixel * (xmin, ymax)
        pixel_buffer = 10
        pxmin = max(0, int(pxmin) - pixel_buffer)
        pymin = max(0, int(pymin) - pixel_buffer)
        pxmax, pymax = to_pixel * (xmax, ymin)
        pxmax = min(raster_template.width, int(pxmax) + pixel_buffer)
        pymax = min(raster_template.height, int(pymax) + pixel_buffer)
        bounds_map[loc_id] = (slice(pymin, pymax), slice(pxmin, pxmax))

    return bounds_map

def build_location_masks(
    block_key: str,
    hiearchy: str,
) -> tuple[dict[str, slice], dict[int, tuple[slice, slice, npt.NDArray[np.bool_]]]]:
    pop_root = Path("/mnt/team/rapidresponse/pub/population-model/modeling/100m/models/2025_03_22.001/raked_predictions/2020q1")
    pop_file = pop_root / f"{block_key}.tif"

    template = rt.load_raster(pop_file)
    template_bbox = get_bbox(template, "EPSG:4326")
    bounds = template_bbox.bounds
    raking_shapes = load_raking_shapes(hiearchy, bounds=bounds)
    raking_shapes = raking_shapes[raking_shapes.intersects(template_bbox)].to_crs(
        template.crs
    )

    shape_values = [
        (shape, loc_id)
        for loc_id, shape in raking_shapes.set_index("location_id")
        .geometry.to_dict()
        .items()
    ]
    bounds_map = build_bounds_map(template, shape_values)

    location_mask = np.zeros_like(template, dtype=np.uint32)
    location_mask = rasterize(
        shape_values,
        out=location_mask,
        transform=template.transform,
        merge_alg=MergeAlg.replace,
    )
    final_bounds_map = {
        location_id: (rows, cols, location_mask[rows, cols] == location_id)
        for location_id, (rows, cols) in bounds_map.items()
    }      
    return bounds, final_bounds_map

def pixel_main(
        years: list[int],
        threshold: float,
        block_key: str,
        hiearchy: str
):
    
    bounds, bounds_map = build_location_masks(block_key, hiearchy)

    result_records = []
    for year in years:
        pop_file = pop_root / f"{year}q1" / f"{block_key}.tif"
        # Load 100m population data for the current year
        pop_100m = rt.load_raster(pop_file)
        # Load 1km population data for the current year
        pop_1km_path = POPULATION_1km_PATH.format(year=year)
        pop_1km = rt.load_raster(pop_1km_path)
        clipped_pop_1km = pop_1km.clip(get_bbox(pop_100m))
        clipped_pop_1km_at_100m = clipped_pop_1km.resample_to(pop_100m, "nearest")
        #
        pop_100m_arr = pop_100m._ndarray  # noqa: SLF001
        pop_1km_arr = clipped_pop_1km_at_100m._ndarray  # noqa: SLF001

        # Create urban/rural classification for the 1km data using the threshold
        pop_1km_urban_simple = np.where(pop_1km_arr >= threshold, 1, 0).astype(float)
        pop_1km_nan_mask = np.isnan(pop_1km_arr)
        pop_1km_urban_simple[pop_1km_nan_mask] = np.nan

        # Calcualte the inflation factor to adjust the 100m population to be 'per 1km' instead of a count
        inflation_factor = 0.01
        pop_100m_as_density = pop_100m_arr.copy()
        pop_100m_as_density = pop_100m_as_density / inflation_factor

        # Create urban/rural classification for the 100m data using same threshold
        pop_100m_urban_simple = np.where(pop_100m_as_density >= threshold, 1, 0).astype(float)
        pop_100m_nan_mask = np.isnan(pop_100m_as_density)
        pop_100m_urban_simple[pop_100m_nan_mask] = np.nan

        # Load population data and grab the underlying ndarray (we don't want the metadata)
        
        pop_raster = rt.load_raster(pop_file)
        pop_arr = pop_raster._ndarray  # noqa: SLF001

        # Replace clim_arr with pop_100m_urban_simple or pop_1km_urban_simple
        weighted_1km_urban_arr_simple = pop_arr * pop_1km_urban_simple  # type: ignore[operator]
        weighted_100m_urban_arr_simple = pop_arr * pop_100m_urban_simple  # type: ignore[operator]

        for location_id, (rows, cols, loc_mask) in bounds_map.items():
            # Subset and mask the weighted climate and population, then sum
            # all non-nan values
            loc_weighted_1km_urban_simple = np.nansum(weighted_1km_urban_arr_simple[rows, cols][loc_mask])
            loc_weighted_100m_urban_simple = np.nansum(weighted_100m_urban_arr_simple[rows, cols][loc_mask])
            loc_pop = np.nansum(pop_arr[rows, cols][loc_mask])

            result_records.append(
                (location_id, year, 
                 loc_weighted_1km_urban_simple, 
                 loc_weighted_100m_urban_simple, 
                 loc_pop)
            )

    results = pd.DataFrame(
        result_records,
        columns=[
            "location_id",
            "year_id",
            f"weighted_1km_urban_threshold_{threshold}_simple", 
            f"weighted_100m_urban_threshold_{threshold}_simple",
            "population",
        ],
    ).sort_values(by=["location_id", "year_id"])
    save_path = DATA_PATH / hiearchy / f"urban_threshold_{threshold}_simple" / block_key
    mkdir(save_path, parents=True, exist_ok=True)
    filename = "000.parquet"
    results.to_parquet(
        save_path / filename,
        index=False,
    )

# Call the function with parsed arguments
pixel_main(years, threshold, block_key, hiearchy)