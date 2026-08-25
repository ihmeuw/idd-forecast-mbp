"""Tests for malaria VE curve construction, validation, and lookup.

The headline test is byte-identity: the curves this repo builds from
VE_ANCHORS.yaml must reproduce, byte for byte, the curves that were handed over
on 2026-08-24. Hashes are pinned here rather than read from shared storage so
the test is fast, hermetic, and cannot silently pass if the reference vintage
moves or is rewritten.
"""
import hashlib

import pandas as pd
import pytest
import yaml

from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.lib.processing import vaccine_efficacy as ve

# sha256 of df.to_csv(index=False) for each cell, verified equal to both the
# received vintage (01-raw_data/.../20260824) and the built processed node.
EXPECTED_SHA256 = {
    "linear_severe0": "be7f656fd7febd41e599f9cc6d79941eb0ab10048ed9d16ac2689287dfc06995",
    "linear_severeSmooth": "651ee4501df110632f9e3800339d68be92f9f4ccc406bcc5260bf4e70c7c8749",
    "loglinear_severe0": "e10de647adb09e204f703d50493ce48d8caeea4e89a5c47342460b886c21bf46",
    "loglinear_severeSmooth": "a0b9b33a22988dec1b02a2cb41e44caa52107077ade7646662e0c5e606abe233",
}
PRODUCTS = ("rtss", "r21")


@pytest.fixture(scope="module")
def anchors() -> dict:
    return yaml.safe_load(rfc.VE_ANCHORS_PATH.read_text())


@pytest.fixture(scope="module")
def cells(anchors) -> dict:
    return ve.build_all_cells(anchors)


def test_all_four_cells_built(cells):
    assert set(cells) == set(EXPECTED_SHA256)


@pytest.mark.parametrize("cell", sorted(EXPECTED_SHA256))
def test_byte_identical_to_received_vintage(cells, cell):
    got = hashlib.sha256(cells[cell].to_csv(index=False).encode()).hexdigest()
    assert got == EXPECTED_SHA256[cell], (
        f"{cell} no longer reproduces the received curve. Either the anchors or the "
        "construction changed -- explain the difference before updating this hash."
    )


def test_every_cell_satisfies_the_loader_contract(cells, anchors):
    sched = anchors["schedule"]
    for df in cells.values():
        ve.validate_ve_frame(df, expected_products=PRODUCTS,
                             dose3_age=sched["dose3_age_months"],
                             booster_age=sched["booster_age_months"])


def test_default_cell_is_one_of_the_built_cells(cells, anchors):
    assert anchors["build"]["default_cell"] in cells


# ---------------------------------------------------------------------------
# validate_ve_frame must reject each way the contract can be broken
# ---------------------------------------------------------------------------
@pytest.fixture
def good(cells) -> pd.DataFrame:
    return cells["loglinear_severe0"].copy()


def test_rejects_missing_column(good):
    with pytest.raises(ValueError, match="missing columns"):
        ve.validate_ve_frame(good.drop(columns=["ve_death_d3"]))


def test_rejects_nulls(good):
    good.loc[good.index[0], "ve_case_d3"] = None
    with pytest.raises(ValueError, match="nulls"):
        ve.validate_ve_frame(good)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_rejects_values_outside_unit_interval(good, value):
    good.loc[good.index[0], "ve_case_d3"] = value
    with pytest.raises(ValueError, match=r"out of \[0,1\]"):
        ve.validate_ve_frame(good)


def test_rejects_missing_product(good):
    with pytest.raises(ValueError, match="missing product"):
        ve.validate_ve_frame(good[good.vaccine == "r21"], expected_products=PRODUCTS)


def test_rejects_gap_in_age_months(good):
    gapped = good[good.age_months != 5]
    with pytest.raises(ValueError, match="gapless run from 0"):
        ve.validate_ve_frame(gapped)


def test_rejects_wrong_dose3_trigger(good):
    """Zeroing month 6 moves the first non-zero dose-3 month later."""
    good.loc[(good.age_months == 6), ["ve_case_d3", "ve_case_d34"]] = 0.0
    with pytest.raises(ValueError, match="dose-3 trigger at month"):
        ve.validate_ve_frame(good)


def test_rejects_wrong_booster_trigger(good):
    """Making the boosted column equal the dose-3 column at 24 pushes the
    detected booster month later."""
    at24 = good.age_months == 24
    good.loc[at24, "ve_case_d34"] = good.loc[at24, "ve_case_d3"].to_numpy()
    with pytest.raises(ValueError, match="booster trigger at month"):
        ve.validate_ve_frame(good)


def test_rejects_curve_that_never_reaches_zero(good):
    last = good.age_months.max()
    good.loc[good.age_months == last, "ve_case_d34"] = 0.2
    with pytest.raises(ValueError, match="does not reach 0"):
        ve.validate_ve_frame(good)


# ---------------------------------------------------------------------------
# VECurve lookup
# ---------------------------------------------------------------------------
@pytest.fixture
def curve(cells) -> ve.VECurve:
    return ve.VECurve.from_frame(cells["loglinear_severe0"])


def test_products_and_month_count(curve, cells):
    assert curve.products() == ("r21", "rtss")
    assert curve.n_months("rtss") == int(cells["loglinear_severe0"].age_months.max()) + 1


def test_zero_before_dose3_and_full_at_trigger(curve):
    assert curve.ve("rtss", "ve_case_d34", 5 / 12) == 0.0
    assert curve.ve("rtss", "ve_case_d34", 0.5) == pytest.approx(0.68)


def test_interpolates_between_whole_months(curve):
    """Half way between months 6 and 7 is the mean of the two."""
    lo = curve.ve("rtss", "ve_case_d3", 6 / 12)
    hi = curve.ve("rtss", "ve_case_d3", 7 / 12)
    assert curve.ve("rtss", "ve_case_d3", 6.5 / 12) == pytest.approx((lo + hi) / 2)


def test_zero_at_and_beyond_the_last_month(curve):
    assert curve.ve("rtss", "ve_case_d3", 100.0) == 0.0
    assert curve.ve("rtss", "ve_case_d3", curve.max_month / 12) == 0.0


def test_unknown_product_or_column_raises(curve):
    with pytest.raises(KeyError, match="no VE curve"):
        curve.ve("nonexistent", "ve_case_d3", 1.0)
    with pytest.raises(KeyError, match="no VE curve"):
        curve.ve("rtss", "ve_nope", 1.0)


def test_delivered_curves_have_no_nonzero_tails(curve):
    """The zero-beyond-the-end lookup rule is only safe while the curves are
    carried out to where VE has actually decayed to zero."""
    assert curve.nonzero_tails() == {}


def test_implied_triggers_match_the_schedule(curve, anchors):
    sched = anchors["schedule"]
    assert curve.implied_trigger_months("rtss") == (sched["dose3_age_months"],
                                                    sched["booster_age_months"])


def test_max_month_defaults_from_the_curves():
    """Direct construction without max_month derives it, so callers building a
    curve by hand (tests, fixtures) do not have to supply it."""
    c = ve.VECurve(curves={"r21": {k: [0.0] * 6 + [0.5] * 6 for k in ve.VE_COLUMNS}})
    assert c.max_month == 11
    assert c.ve("r21", "ve_case_d3", 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("age_years", [0.0, -1.0])
def test_age_at_or_below_zero_returns_the_month_zero_value(curve, age_years):
    """Nobody is protected at birth, so month 0 is 0 -- but the lookup returns the
    curve's own first value rather than assuming it."""
    assert curve.ve("rtss", "ve_case_d3", age_years) == pytest.approx(0.0)


def test_load_ve_curve_round_trips_a_written_file(cells, anchors, tmp_path):
    """The public loader: read a CSV, validate it, return a usable curve."""
    sched = anchors["schedule"]
    path = tmp_path / "ve_loglinear_severe0.csv"
    cells["loglinear_severe0"].to_csv(path, index=False)

    loaded = ve.load_ve_curve(str(path), expected_products=PRODUCTS,
                              dose3_age=sched["dose3_age_months"],
                              booster_age=sched["booster_age_months"])
    assert loaded.products() == ("r21", "rtss")
    assert loaded.ve("rtss", "ve_case_d34", 0.5) == pytest.approx(0.68)


def test_load_ve_curve_rejects_an_invalid_file(cells, tmp_path):
    """Validation happens at load, so a bad file never becomes a curve."""
    bad = cells["loglinear_severe0"].drop(columns=["ve_death_d34"])
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        ve.load_ve_curve(str(path))
