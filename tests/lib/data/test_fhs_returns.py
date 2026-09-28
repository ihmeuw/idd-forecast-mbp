"""The FHS return reader: the round map, the file contract, and one-draw reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
import yaml

from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.lib.data import fhs_returns as fr

if TYPE_CHECKING:
    from pathlib import Path

LOCS = [1, 4, 5]
YEARS = [2022, 2023, 2024]
AGES = [2, 3, 22]
SEXES = [1, 2, 3]
DRAWS = [0, 1, 2]


def _return_ds(*, scenarios: int = 1) -> xr.Dataset:
    rng = np.random.default_rng(0)
    shape = (len(DRAWS), scenarios, len(AGES), len(YEARS), len(LOCS), len(SEXES))
    values = rng.uniform(1, 100, shape)
    return xr.Dataset(
        {
            fr.VALUE_VAR: (
                (
                    "draw",
                    "scenario",
                    "age_group_id",
                    "year_id",
                    "location_id",
                    "sex_id",
                ),
                values,
            )
        },
        coords={
            "draw": DRAWS,
            "scenario": list(range(scenarios)),
            "age_group_id": AGES,
            "year_id": YEARS,
            "location_id": LOCS,
            "sex_id": SEXES,
        },
    )


@pytest.fixture
def return_file(tmp_path: Path) -> Path:
    p = tmp_path / "death" / "ds_death" / "malaria.nc"
    p.parent.mkdir(parents=True)
    _return_ds().to_netcdf(p)
    return p


@pytest.fixture
def map_file(tmp_path: Path) -> Path:
    p = tmp_path / "fhs_returns.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "cause_file": "malaria.nc",
                "datasets": {
                    "ssp245": {"death": "ds_death", "incidence": "ds_inc"},
                    "ssp126": {"death": "ds_death_126"},
                },
            }
        )
    )
    return p


# ------------------------------------------------------------------ the map
def test_map_paths_and_lookups(map_file: Path, tmp_path: Path) -> None:
    m = fr.load_fhs_return_map(map_file, root=tmp_path)
    assert m.root == tmp_path
    assert m.path("ssp245", "death") == tmp_path / "death" / "ds_death" / "malaria.nc"
    assert (
        m.path("ssp126", "death") == tmp_path / "death" / "ds_death_126" / "malaria.nc"
    )
    assert m.ssps() == ("ssp126", "ssp245")
    assert m.measures() == ("death", "incidence")
    with pytest.raises(ValueError, match="no FHS dataset"):
        m.path("ssp585", "death")


def test_map_root_precedence(map_file: Path, tmp_path: Path) -> None:
    assert fr.load_fhs_return_map(map_file).root == mbpc.FHS_RESULTS_PATH
    raw = yaml.safe_load(map_file.read_text())
    raw["root"] = str(tmp_path / "from_file")
    map_file.write_text(yaml.safe_dump(raw))
    assert fr.load_fhs_return_map(map_file).root == tmp_path / "from_file"
    assert (
        fr.load_fhs_return_map(map_file, root=tmp_path / "arg").root == tmp_path / "arg"
    )


def test_map_missing_file_refuses(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fr.load_fhs_return_map(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("raw", "exc", "match"),
    [
        (["not", "a", "mapping"], TypeError, "mapping at the top level"),
        ({"datasets": {}}, ValueError, "keys must be"),
        (
            {"cause_file": "m.nc", "datasets": {}, "extra": 1},
            ValueError,
            "keys must be",
        ),
        ({"cause_file": 3, "datasets": {}}, TypeError, "must be a string"),
        (
            {"cause_file": "m.nc", "datasets": {"ssp245": "flat"}},
            TypeError,
            "must map measure",
        ),
        (
            {"cause_file": "m.nc", "datasets": {"ssp245": {"death": ""}}},
            ValueError,
            "non-empty string",
        ),
        ({"cause_file": "m.nc", "datasets": {}}, ValueError, "datasets is empty"),
    ],
)
def test_map_shape_refusals(
    tmp_path: Path, raw: object, exc: type[Exception], match: str
) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(exc, match=match):
        fr.load_fhs_return_map(p)


# ------------------------------------------------------------------ the file contract
def test_validate_accepts_the_return_shape() -> None:
    fr.validate_fhs_return(_return_ds())


def test_validate_refuses_wrong_dims_missing_value_and_two_scenarios() -> None:
    with pytest.raises(ValueError, match="dims are"):
        fr.validate_fhs_return(_return_ds().rename({"sex_id": "sex"}))
    with pytest.raises(ValueError, match="no 'value' variable"):
        fr.validate_fhs_return(_return_ds().rename_vars({fr.VALUE_VAR: "draws"}))
    with pytest.raises(ValueError, match="2 scenarios"):
        fr.validate_fhs_return(_return_ds(scenarios=2))


def test_validate_refuses_non_integer_ids() -> None:
    ds = _return_ds().assign_coords(location_id=[1.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="not integer"):
        fr.validate_fhs_return(ds)


# ------------------------------------------------------------------ reads
def test_available_draws(return_file: Path) -> None:
    assert fr.available_draws(return_file) == DRAWS


def test_read_one_draw_subsets_and_types(return_file: Path) -> None:
    frame = fr.read_fhs_return_draw(
        return_file, 1, age_group_ids=[2, 3], sex_ids=[1, 2], years=[2023, 2024]
    )
    assert list(frame.columns) == list(fr.FRAME_COLUMNS)
    assert len(frame) == len(LOCS) * 2 * 2 * 2
    assert {str(frame[c].dtype) for c in fr.FRAME_COLUMNS[:-1]} == {"int64"}
    assert str(frame[fr.VALUE_VAR].dtype) == "float64"
    assert set(frame.age_group_id) == {2, 3}
    assert set(frame.sex_id) == {1, 2}
    assert set(frame.year_id) == {2023, 2024}
    expected = (
        _return_ds()[fr.VALUE_VAR]
        .sel(draw=1, scenario=0, location_id=4, year_id=2024, age_group_id=3, sex_id=2)
        .item()
    )
    got = frame[
        (frame.location_id == 4)
        & (frame.year_id == 2024)
        & (frame.age_group_id == 3)
        & (frame.sex_id == 2)
    ]
    assert got[fr.VALUE_VAR].item() == pytest.approx(expected)


def test_read_all_years_when_none_given(return_file: Path) -> None:
    frame = fr.read_fhs_return_draw(return_file, 0, age_group_ids=AGES, sex_ids=SEXES)
    assert set(frame.year_id) == set(YEARS)
    assert len(frame) == len(LOCS) * len(YEARS) * len(AGES) * len(SEXES)


def test_read_refuses_unknown_draw_and_absent_ids(return_file: Path) -> None:
    with pytest.raises(ValueError, match="draw 7 not in file"):
        fr.read_fhs_return_draw(return_file, 7, age_group_ids=[2], sex_ids=[1])
    with pytest.raises(
        ValueError, match="requested age_group_id not in file: \\[99\\]"
    ):
        fr.read_fhs_return_draw(return_file, 0, age_group_ids=[2, 99], sex_ids=[1])
    with pytest.raises(ValueError, match="requested year_id not in file"):
        fr.read_fhs_return_draw(
            return_file, 0, age_group_ids=[2], sex_ids=[1], years=[2100]
        )
