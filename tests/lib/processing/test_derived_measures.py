"""The FHS measure table: sources, sums, and the refusals that keep it consistent."""

from __future__ import annotations

import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing import derived_measures as dm


def test_table_is_the_first_submission_chain() -> None:
    assert {d.output: d.source for d in dm.FHS_RAKED} == {
        "incidence": "incidence",
        "death": "mortality",
        "yll": "mortality",
        "yld": "incidence",
    }
    assert dm.FHS_SUMMED == {"daly": ("yll", "yld")}
    assert dm.all_fhs_measures() == ("incidence", "death", "yll", "yld", "daly")


def test_lookups() -> None:
    assert dm.source_measure("yll") == "mortality"
    assert dm.source_measure("yld") == "incidence"
    assert dm.raked_outputs_of("mortality") == ("death", "yll")
    assert dm.raked_outputs_of("incidence") == ("incidence", "yld")
    assert dm.components_of("daly") == ("yll", "yld")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: dm.source_measure("daly"), "not a raked FHS measure"),
        (lambda: dm.raked_outputs_of("daly"), "not one of our measures"),
        (lambda: dm.components_of("yll"), "not a summed FHS measure"),
    ],
)
def test_lookup_refusals(call: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("raked", "summed", "match"),
    [
        (
            (
                dm.MeasureDerivation("a", "incidence"),
                dm.MeasureDerivation("a", "mortality"),
            ),
            {},
            "duplicate",
        ),
        ((dm.MeasureDerivation("a", "prevalence"),), {}, "not one of our measures"),
        (
            (dm.MeasureDerivation("a", "incidence"),),
            {"a": ("a",)},
            "both raked and summed",
        ),
        ((dm.MeasureDerivation("a", "incidence"),), {"b": ("zzz",)}, "names parts"),
        ((dm.MeasureDerivation("a", "incidence"),), {"b": ()}, "names parts"),
    ],
)
def test_validate_table_refusals(
    raked: tuple[dm.MeasureDerivation, ...],
    summed: dict[str, tuple[str, ...]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        dm.validate_table(raked, summed)


def _frame(values: dict[tuple[int, int], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"location_id": loc, "year_id": year, "value": v}
            for (loc, year), v in values.items()
        ]
    )


def test_sum_components_aligns_outer_and_fills_absent_cells_with_zero() -> None:
    yll = _frame({(1, 2023): 3.0, (1, 2024): 4.0, (2, 2023): 5.0})
    yld = _frame({(1, 2023): 0.5, (2, 2024): 1.0})
    out = dm.sum_components(
        {"yll": yll, "yld": yld},
        "daly",
        keys=["location_id", "year_id"],
        value_col="value",
    )
    got = out.set_index(["location_id", "year_id"])["value"].to_dict()
    assert got == {(1, 2023): 3.5, (1, 2024): 4.0, (2, 2023): 5.0, (2, 2024): 1.0}


def test_sum_components_refusals() -> None:
    yll = _frame({(1, 2023): 3.0})
    with pytest.raises(ValueError, match="needs component frame"):
        dm.sum_components(
            {"yll": yll}, "daly", keys=["location_id", "year_id"], value_col="value"
        )
    with pytest.raises(ValueError, match="lacks column"):
        dm.sum_components(
            {"yll": yll, "yld": yll.drop(columns="value")},
            "daly",
            keys=["location_id", "year_id"],
            value_col="value",
        )
