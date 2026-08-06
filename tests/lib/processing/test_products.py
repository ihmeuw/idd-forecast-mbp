"""Tests for the product contract and the anchor diagnostic.

The two behaviours worth defending hardest, because getting either wrong is silent:

* a rate must divide by *that level's own* population row, so the validator re-derives it
* a window-mean anchor must be compared against its window mean, not against observed at a
  single year -- checked the wrong way, a correct dengue run looks catastrophic and a broken
  one can look fine
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.cause_spec import (
    AnchorKind,
    AnchorSpec,
    MeasureSpec,
    MeasureStructure,
    get_cause_spec,
)
from idd_forecast_mbp.lib.processing.products import (
    BASE_COLUMNS,
    anchor_diagnostic,
    expected_columns,
    measure_columns,
    validate_products,
)


@pytest.fixture
def measures() -> MeasureStructure:
    return MeasureStructure(
        (
            MeasureSpec("incidence", "inc_count", "inc_rate"),
            MeasureSpec("mortality", "mort_count", "mort_rate"),
        )
    )


@pytest.fixture
def malaria():
    return get_cause_spec("malaria")


def _product(n_loc: int = 3, *, population: float = 1000.0) -> pd.DataFrame:
    """A minimal contract-conforming product: rates exactly count/population.

    Counts grow faster than population on purpose, so the per-location rates genuinely differ.
    With counts proportional to population every rate is identical and any test of a wrong
    denominator becomes vacuous.
    """
    rows = []
    for i in range(n_loc):
        pop = population * (i + 1)
        inc, mort = 100.0 * (i + 1) ** 2, 5.0 * (i + 1) ** 2
        row = {
            "location_id": 10 + i,
            "year_id": 2050,
            "level": 3,
            "population": pop,
        }
        for base, mean in (("inc_count", inc), ("mort_count", mort)):
            row[f"{base}_mean"] = mean
            row[f"{base}_lower"] = mean * 0.9
            row[f"{base}_upper"] = mean * 1.1
        for base, mean in (("inc_rate", inc), ("mort_rate", mort)):
            row[f"{base}_mean"] = mean / pop
            row[f"{base}_lower"] = mean * 0.9 / pop
            row[f"{base}_upper"] = mean * 1.1 / pop
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("location_id", "year_id", "level"):
        df[c] = df[c].astype("int64")
    return df


# --------------------------------------------------------------------------- schema


def test_expected_columns_covers_base_and_every_measure(measures):
    cols = expected_columns(measures)
    assert set(BASE_COLUMNS) <= set(cols)
    for base in ("inc_count", "inc_rate", "mort_count", "mort_rate"):
        for stat in ("mean", "lower", "upper"):
            assert f"{base}_{stat}" in cols


def test_intermediate_measures_are_not_part_of_the_contract():
    """CFR is computed but not delivered, so it must not be a required column."""
    ms = MeasureStructure(
        (
            MeasureSpec("cfr", None, None, intermediate=True),
            MeasureSpec("incidence", "inc_count", "inc_rate"),
        )
    )
    assert not any("cfr" in c for c in measure_columns(ms))


# --------------------------------------------------------------------------- happy path


def test_conforming_product_validates(malaria, measures):
    rep = validate_products(_product(), malaria, measures)
    assert rep.ok, rep.summary()
    rep.raise_if_failed()


def test_missing_columns_reported_and_short_circuits(malaria, measures):
    df = _product().drop(columns=["mort_rate_upper"])
    rep = validate_products(df, malaria, measures)
    assert not rep.ok
    assert any("missing required columns" in p for p in rep.problems)


def test_empty_product_fails(malaria, measures):
    rep = validate_products(_product().iloc[0:0], malaria, measures)
    assert not rep.ok
    assert any("empty" in p for p in rep.problems)


# --------------------------------------------------------------------------- the rate invariant


def test_rate_derived_from_summed_child_population_is_caught(malaria, measures):
    """The denominator error this validator exists for.

    Dividing by a sum of children's populations instead of the level's own row gives a rate that
    is too small; nothing about the file looks wrong unless the rate is re-derived.
    """
    df = _product()
    wrong_denominator = df.population.sum()
    df["inc_rate_mean"] = df.inc_count_mean / wrong_denominator
    rep = validate_products(df, malaria, measures)
    assert not rep.ok
    assert any("rate != count / own population" in p for p in rep.problems)
    assert any("never a sum of children" in p for p in rep.problems)


def test_population_weighted_average_of_child_rates_is_caught(malaria, measures):
    df = _product()
    df["mort_rate_mean"] = float(np.average(df.mort_rate_mean, weights=df.population))
    rep = validate_products(df, malaria, measures)
    assert not rep.ok
    assert any("mortality mean" in p for p in rep.problems)


def test_rate_check_skips_zero_population_rows_instead_of_dividing_by_zero(malaria, measures):
    """162 admin-2 units have population exactly 0; that must not crash or falsely fail."""
    df = _product()
    df.loc[0, "population"] = 0.0
    for base in ("inc", "mort"):
        for stat in ("mean", "lower", "upper"):
            df.loc[0, f"{base}_count_{stat}"] = 0.0
            df.loc[0, f"{base}_rate_{stat}"] = 0.0
    rep = validate_products(df, malaria, measures)
    assert rep.ok, rep.summary()
    assert any("population == 0" in n for n in rep.notes)


# --------------------------------------------------------------------------- other checks


def test_float_id_columns_are_a_bug(malaria, measures):
    df = _product()
    df["location_id"] = df.location_id.astype("float64")
    rep = validate_products(df, malaria, measures)
    assert any("must be an integer dtype" in p for p in rep.problems)


def test_draw_axis_must_be_collapsed(malaria, measures):
    df = _product()
    df["draw"] = 0
    rep = validate_products(df, malaria, measures)
    assert any("draws must be collapsed" in p for p in rep.problems)


def test_redundant_constant_dimension_column_is_rejected(malaria, measures):
    """ssp belongs in the filename, not as a column with one value in every row."""
    df = _product()
    df["ssp_scenario"] = "ssp245"
    rep = validate_products(df, malaria, measures)
    assert any("belongs in the filename" in p for p in rep.problems)


def test_a_varying_dimension_column_is_not_flagged(malaria, measures):
    """Only a SINGLE-valued dimension column is redundant."""
    df = pd.concat([_product(), _product()], ignore_index=True)
    df["trajectory"] = ["Baseline"] * 3 + ["Constant"] * 3
    df["location_id"] = range(len(df))
    rep = validate_products(df, malaria, measures)
    assert not any("belongs in the filename" in p for p in rep.problems)


def test_duplicate_location_year_rows_rejected(malaria, measures):
    df = pd.concat([_product(1), _product(1)], ignore_index=True)
    rep = validate_products(df, malaria, measures)
    assert any("duplicated" in p for p in rep.problems)


def test_inverted_interval_rejected(malaria, measures):
    df = _product()
    df.loc[0, "inc_count_lower"] = df.loc[0, "inc_count_mean"] * 2
    rep = validate_products(df, malaria, measures)
    assert any("lower exceeds mean" in p for p in rep.problems)


def test_negative_count_rejected(malaria, measures):
    df = _product()
    df.loc[0, "mort_count_mean"] = -1.0
    df.loc[0, "mort_rate_mean"] = -1.0 / df.loc[0, "population"]
    rep = validate_products(df, malaria, measures)
    assert any("negative value" in p for p in rep.problems)


def test_negative_population_rejected(malaria, measures):
    df = _product()
    df.loc[0, "population"] = -5.0
    rep = validate_products(df, malaria, measures)
    assert any("negative population" in p for p in rep.problems)


def test_expected_levels_mismatch_reported(malaria, measures):
    rep = validate_products(_product(), malaria, measures, expected_levels=[0, 1, 2, 3])
    assert any("levels present" in p for p in rep.problems)


def test_absent_means_zero_is_surfaced_as_a_note_for_dengue(measures):
    dengue = get_cause_spec("dengue")
    rep = validate_products(_product(), dengue, measures)
    assert rep.ok
    assert any("mean ZERO, not unknown" in n for n in rep.notes)


def test_additivity_skipped_without_hierarchy_and_noted(malaria, measures):
    rep = validate_products(_product(), malaria, measures)
    assert any("additivity not checked" in n for n in rep.notes)


def test_count_additivity_catches_a_parent_that_is_not_the_sum_of_children(malaria, measures):
    child = _product(2)
    child["location_id"] = [11, 12]
    parent = _product(1)
    parent["location_id"] = [1]
    parent["level"] = 2
    # Correct parent = 100 + 200 = 300; make it wrong.
    pop = float(child.population.sum())
    parent["population"] = pop
    for stat, mult in (("mean", 1.0), ("lower", 0.9), ("upper", 1.1)):
        parent[f"inc_count_{stat}"] = 999.0 * mult
        parent[f"inc_rate_{stat}"] = 999.0 * mult / pop
        tot = float(child[f"mort_count_{stat}"].sum())
        parent[f"mort_count_{stat}"] = tot
        parent[f"mort_rate_{stat}"] = tot / pop
    df = pd.concat([parent, child], ignore_index=True)
    hierarchy = pd.DataFrame({"location_id": [1, 11, 12], "parent_id": [0, 1, 1]})
    rep = validate_products(df, malaria, measures, hierarchy=hierarchy)
    assert any("parent count != sum of children" in p for p in rep.problems)
    assert not any("mortality: parent count" in p for p in rep.problems)


def test_raise_if_failed_lists_every_problem(malaria, measures):
    df = _product()
    df["location_id"] = df.location_id.astype("float64")
    df.loc[0, "population"] = -1.0
    rep = validate_products(df, malaria, measures)
    with pytest.raises(ValueError, match="violates the product contract"):
        rep.raise_if_failed()
    assert len(rep.problems) >= 2


# --------------------------------------------------------------------------- anchor diagnostic


def _observed_spike() -> pd.DataFrame:
    """A decade where the final year is an epidemic spike -- the dengue 2023 situation."""
    years = list(range(2014, 2024))
    vals = [9.0, 10.0, 12.0, 15.0, 9.3, 30.6, 18.0, 20.0, 22.0, 37.5]
    return pd.DataFrame(
        {"location_id": [1] * len(years), "year_id": years, "value": vals}
    )


def test_point_anchor_passes_when_it_reproduces_observed():
    obs = _observed_spike()
    pred = pd.DataFrame({"location_id": [1], "year_id": [2023], "value": [37.5]})
    rep = anchor_diagnostic(
        pred, obs, AnchorSpec(AnchorKind.POINT, (2023,)), value_col="value"
    )
    assert rep.passed
    assert not rep.flagged
    assert "reproduces observed" in rep.message


def test_point_anchor_fails_loudly_when_it_does_not():
    obs = _observed_spike()
    pred = pd.DataFrame({"location_id": [1], "year_id": [2023], "value": [20.3]})
    rep = anchor_diagnostic(
        pred, obs, AnchorSpec(AnchorKind.POINT, (2023,)), value_col="value"
    )
    assert not rep.passed
    assert rep.flagged
    assert "DOES NOT reproduce observed" in rep.message


def test_window_mean_anchor_accepts_a_run_far_from_the_final_year():
    """Dengue F4: ~20.3 predicted against a ~18.3 decade mean passes; vs observed 2023 it would not.

    This is the case that must not be reported as a failure.
    """
    obs = _observed_spike()
    window_mean = obs.value.mean()
    pred = pd.DataFrame(
        {"location_id": [1], "year_id": [2023], "value": [window_mean * 1.04]}
    )
    anchor = AnchorSpec(AnchorKind.WINDOW_MEAN, tuple(range(2014, 2024)))
    rep = anchor_diagnostic(pred, obs, anchor, value_col="value")
    assert not rep.flagged
    assert "Exactness is NOT expected" in rep.message
    # And the prediction is far from observed 2023, which the old check would have flagged.
    assert abs(pred.value.iloc[0] / 37.5 - 1) > 0.4


def test_window_mean_anchor_flags_a_real_level_error():
    """F4 mortality: +21% over its own anchor target is worth investigating."""
    obs = _observed_spike()
    pred = pd.DataFrame(
        {"location_id": [1], "year_id": [2023], "value": [obs.value.mean() * 1.21]}
    )
    anchor = AnchorSpec(AnchorKind.WINDOW_MEAN, tuple(range(2014, 2024)))
    rep = anchor_diagnostic(pred, obs, anchor, value_col="value")
    assert rep.flagged
    assert "worth investigating" in rep.message
    assert not rep.passed


def test_window_mean_never_claims_passed_even_when_close():
    """passed is only meaningful for an anchor that guarantees exactness."""
    obs = _observed_spike()
    pred = pd.DataFrame({"location_id": [1], "year_id": [2023], "value": [obs.value.mean()]})
    rep = anchor_diagnostic(
        pred, obs, AnchorSpec(AnchorKind.WINDOW_MEAN, tuple(range(2014, 2024))),
        value_col="value",
    )
    assert rep.passed is False
    assert rep.flagged is False


def test_anchor_diagnostic_reports_the_ratio_table():
    obs = _observed_spike()
    pred = pd.DataFrame({"location_id": [1], "year_id": [2023], "value": [20.0]})
    rep = anchor_diagnostic(
        pred, obs, AnchorSpec(AnchorKind.WINDOW_MEAN, tuple(range(2014, 2024))),
        value_col="value",
    )
    assert {"anchor_target", "predicted", "ratio", "rel_diff", "n_years"} <= set(rep.table.columns)
    assert rep.table.n_years.iloc[0] == 10


def test_anchor_diagnostic_raises_when_the_prediction_year_is_absent():
    obs = _observed_spike()
    pred = pd.DataFrame({"location_id": [1], "year_id": [2099], "value": [1.0]})
    with pytest.raises(ValueError, match="no predicted rows at year 2023"):
        anchor_diagnostic(pred, obs, AnchorSpec(AnchorKind.POINT, (2023,)), value_col="value")


def test_anchor_diagnostic_raises_when_no_groups_overlap():
    obs = _observed_spike()
    pred = pd.DataFrame({"location_id": [999], "year_id": [2023], "value": [1.0]})
    with pytest.raises(ValueError, match="no groups shared"):
        anchor_diagnostic(pred, obs, AnchorSpec(AnchorKind.POINT, (2023,)), value_col="value")
