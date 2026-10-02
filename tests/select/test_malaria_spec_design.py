"""Tests for idd_forecast_mbp.select.malaria_spec_design (the per-run typed spec design)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from idd_forecast_mbp.select.malaria_spec_design import (
    CURRENT_DESIGN,
    DESIGN_FILE,
    OUT,
    SpecDesign,
    load_design,
    write_design,
)

if TYPE_CHECKING:
    from pathlib import Path

TOY = SpecDesign(
    {
        "a": [("x", "linear", None), ("x", "mpi", 4)],
        "b": [("y", "smooth", None)],
    }
)


def test_universe_is_the_full_factorial() -> None:
    assert len(TOY.universe.configs) == 3 * 2
    assert TOY.axes == ("a", "b")
    # configs[0] is spec_index 1: every axis out
    assert all(v == OUT for v in TOY.universe.configs[0].values())


@pytest.mark.parametrize(
    ("config", "formula", "n_smooths", "n_scams", "n_terms"),
    [
        ({"a": OUT, "b": OUT}, "logit_malaria_pfpr ~ A0_af", 0, 0, 0),
        ({"a": "x|linear", "b": OUT}, "logit_malaria_pfpr ~ x + A0_af", 0, 0, 1),
        (
            {"a": "x|mpi", "b": "y|smooth"},
            'logit_malaria_pfpr ~ s(x, k = 4, bs = "mpi") + s(y, k = 6) + A0_af',
            2,
            1,
            2,
        ),
    ],
)
def test_formula_and_counts(
    config: dict[str, str], formula: str, n_smooths: int, n_scams: int, n_terms: int
) -> None:
    assert TOY.formula_text(config) == formula
    assert TOY.n_smooths(config) == n_smooths
    assert TOY.n_scams(config) == n_scams
    assert TOY.n_terms(config) == n_terms


def test_current_design_size_and_maxima() -> None:
    configs = CURRENT_DESIGN.universe.configs
    assert len(configs) == 8100
    assert max(CURRENT_DESIGN.n_smooths(c) for c in configs) == 8
    assert max(CURRENT_DESIGN.n_scams(c) for c in configs) == 6


def test_json_round_trip_keeps_groups_and_order() -> None:
    back = SpecDesign.from_json(CURRENT_DESIGN.to_json())
    assert back.groups == CURRENT_DESIGN.groups
    assert back.axes == CURRENT_DESIGN.axes
    assert back.universe.configs == CURRENT_DESIGN.universe.configs


def test_write_then_load_design(tmp_path: Path) -> None:
    path = write_design(TOY, tmp_path)
    assert path == tmp_path / DESIGN_FILE
    assert load_design(tmp_path).groups == TOY.groups


def test_write_design_refuses_to_replace(tmp_path: Path) -> None:
    write_design(TOY, tmp_path)
    with pytest.raises(FileExistsError):
        write_design(CURRENT_DESIGN, tmp_path)


def test_load_design_without_file_refuses(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="predates per-run designs"):
        load_design(tmp_path)
