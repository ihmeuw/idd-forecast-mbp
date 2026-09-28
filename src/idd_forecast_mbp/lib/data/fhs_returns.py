"""Reader for the FHS returns: what FHS sends back for the incidence and mortality we submitted.

One netCDF per (measure, dataset), ``<root>/<measure>/<dataset>/<cause_file>``, with dims
``draw x scenario x age_group_id x year_id x location_id x sex_id`` and one variable ``value``
(counts). Which dataset holds which (SSP, measure) is a fact about one FHS round, so it lives
in a small YAML the caller names, never in code; the root defaults to
``constants.FHS_RESULTS_PATH``. A read is one draw, subset to the age groups, sexes and years
the caller is raking, returned as a long frame with integer ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.io.netcdf import read_netcdf_with_integer_ids

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import xarray as xr

FHS_DIMS: frozenset[str] = frozenset(
    {"draw", "scenario", "age_group_id", "year_id", "location_id", "sex_id"}
)
VALUE_VAR = "value"
ID_COORDS: tuple[str, ...] = (
    "location_id",
    "year_id",
    "age_group_id",
    "sex_id",
    "draw",
)
FRAME_COLUMNS: tuple[str, ...] = (
    "location_id",
    "year_id",
    "age_group_id",
    "sex_id",
    VALUE_VAR,
)
_MAP_KEYS: frozenset[str] = frozenset({"cause_file", "datasets", "root"})
_MAX_LISTED = 10  # absent ids named in an error before truncating


@dataclass(frozen=True)
class FhsReturnMap:
    """Where one FHS round's returns are: root, the per-cause file name, and dataset per (SSP, measure)."""

    root: Path
    cause_file: str
    datasets: Mapping[tuple[str, str], str]

    def path(self, ssp: str, measure: str) -> Path:
        """``<root>/<measure>/<dataset>/<cause_file>`` for (ssp, measure); refuses an unknown pair."""
        try:
            dataset = self.datasets[(ssp, measure)]
        except KeyError:
            msg = f"no FHS dataset for ({ssp!r}, {measure!r}); known: {sorted(self.datasets, key=str)}"
            raise ValueError(msg) from None
        return self.root / measure / dataset / self.cause_file

    def ssps(self) -> tuple[str, ...]:
        return tuple(sorted({s for s, _ in self.datasets}))

    def measures(self) -> tuple[str, ...]:
        return tuple(sorted({m for _, m in self.datasets}))


def _parse_datasets(raw: dict[Any, Any], path: Path) -> dict[tuple[str, str], str]:
    """``{ssp: {measure: dataset}}`` -> ``{(ssp, measure): dataset}``; refuses empty or non-string entries."""
    datasets: dict[tuple[str, str], str] = {}
    for ssp, per_measure in raw.items():
        if not isinstance(per_measure, dict):
            msg = f"{path}: datasets[{ssp!r}] must map measure -> dataset"
            raise TypeError(msg)
        for measure, dataset in per_measure.items():
            if not isinstance(dataset, str) or not dataset:
                msg = (
                    f"{path}: datasets[{ssp!r}][{measure!r}] must be a non-empty string"
                )
                raise ValueError(msg)
            datasets[(str(ssp), str(measure))] = dataset
    if not datasets:
        msg = f"{path}: datasets is empty"
        raise ValueError(msg)
    return datasets


def load_fhs_return_map(
    yaml_path: str | Path, *, root: Path | None = None
) -> FhsReturnMap:
    """Read the round's YAML: ``cause_file``, ``datasets: {ssp: {measure: dataset}}``, optional ``root``.

    ``root`` given here wins over the file's, which wins over ``constants.FHS_RESULTS_PATH``.
    Unknown or missing keys and non-string values refuse.
    """
    path = Path(yaml_path)
    if not path.is_file():
        msg = f"FHS return map not found: {path}"
        raise FileNotFoundError(msg)
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        msg = f"{path}: expected a mapping at the top level"
        raise TypeError(msg)
    unknown = set(raw) - _MAP_KEYS
    if unknown or "cause_file" not in raw or "datasets" not in raw:
        msg = f"{path}: keys must be cause_file, datasets[, root]; got {sorted(raw)}"
        raise ValueError(msg)
    if not isinstance(raw["cause_file"], str) or not isinstance(raw["datasets"], dict):
        msg = f"{path}: cause_file must be a string and datasets a mapping"
        raise TypeError(msg)
    datasets = _parse_datasets(raw["datasets"], path)
    if root is not None:
        resolved_root = Path(root)
    elif "root" in raw:
        resolved_root = Path(str(raw["root"]))
    else:
        resolved_root = mbpc.FHS_RESULTS_PATH
    return FhsReturnMap(
        root=resolved_root, cause_file=raw["cause_file"], datasets=datasets
    )


def validate_fhs_return(ds: xr.Dataset) -> None:
    """Refuse a dataset that is not shaped like an FHS return (dims, the value variable, one scenario, integer ids)."""
    dims = {str(d) for d in ds.dims}
    if dims != FHS_DIMS:
        msg = f"FHS return dims are {sorted(dims)}; expected {sorted(FHS_DIMS)}"
        raise ValueError(msg)
    if VALUE_VAR not in ds.data_vars:
        msg = f"FHS return has no {VALUE_VAR!r} variable; has {list(ds.data_vars)}"
        raise ValueError(msg)
    if ds.sizes["scenario"] != 1:
        msg = f"FHS return carries {ds.sizes['scenario']} scenarios; one file holds one scenario"
        raise ValueError(msg)
    for coord in ID_COORDS:
        if not pd.api.types.is_integer_dtype(ds[coord].dtype):
            msg = f"FHS return coordinate {coord!r} is {ds[coord].dtype}, not integer"
            raise ValueError(msg)


def available_draws(path: str | Path) -> list[int]:
    """The draw ids a return file holds (metadata only)."""
    with read_netcdf_with_integer_ids(path) as ds:
        validate_fhs_return(ds)
        return [int(d) for d in ds["draw"].to_numpy()]


def read_fhs_return_draw(
    path: str | Path,
    draw: int,
    *,
    age_group_ids: Sequence[int],
    sex_ids: Sequence[int],
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """One draw of one return, at the requested age groups, sexes and years, as a long frame.

    Every requested id must be in the file (a silent subset would rake to a partial target);
    ``years`` None means every year the file holds. Returns
    ``[location_id, year_id, age_group_id, sex_id, value]`` with int64 ids and float values.
    """
    with read_netcdf_with_integer_ids(path) as ds:
        validate_fhs_return(ds)
        have_draws = {int(d) for d in ds["draw"].to_numpy()}
        if int(draw) not in have_draws:
            msg = f"{path}: draw {draw} not in file (has {len(have_draws)} draws, {min(have_draws)}..{max(have_draws)})"
            raise ValueError(msg)
        wanted: dict[str, list[int]] = {
            "age_group_id": [int(a) for a in age_group_ids],
            "sex_id": [int(s) for s in sex_ids],
        }
        if years is not None:
            wanted["year_id"] = [int(y) for y in years]
        for coord, ids in wanted.items():
            have = {int(v) for v in ds[coord].to_numpy()}
            absent = sorted(set(ids) - have)
            if absent:
                msg = f"{path}: requested {coord} not in file: {absent[:_MAX_LISTED]}{'...' if len(absent) > _MAX_LISTED else ''}"
                raise ValueError(msg)
        slab = (
            ds[VALUE_VAR]
            .sel(draw=int(draw))
            .squeeze("scenario", drop=True)
            .sel(indexers=wanted)
        )
        frame = slab.to_dataframe(name=VALUE_VAR).reset_index()
    frame = frame[list(FRAME_COLUMNS)]
    for col in FRAME_COLUMNS[:-1]:
        frame[col] = frame[col].astype("int64")
    frame[VALUE_VAR] = frame[VALUE_VAR].astype("float64")
    return frame.reset_index(drop=True)
