"""Tests for the pure presentation helpers behind the vaccine impact figures."""
import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.viz.vaccine_impact import (
    PRODUCT_PRETTY,
    PRODUCT_SHORT,
    VARIANT_PRETTY,
    VARIANT_SHORT,
    _ascii,
    _box_stats,
    _cap,
    _lighten,
    _normalize_vaccine,
    _normalize_variants,
    _pair_labels,
    _ssp_color,
    _ssp_label,
    _summarize_generic,
)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("a · b", "a | b"),
    ("a — b", "a - b"),
    ("a – b", "a - b"),
    ("plain ascii", "plain ascii"),
])
def test_ascii_folds_typographic_characters(raw, expected):
    """The table is read in a spreadsheet; UTF-8 punctuation showed up as mojibake."""
    assert _ascii(raw) == expected


def test_ascii_drops_anything_it_cannot_map():
    assert _ascii("café ☃") == "caf "


def test_cap_sentence_cases_without_touching_the_rest():
    assert _cap("malaria cases averted") == "Malaria cases averted"
    assert _cap("") == ""
    assert _cap("R21 everywhere") == "R21 everywhere"


def test_ssp_label_and_colour_fall_back_for_unknown_scenarios():
    assert _ssp_label("ssp245") == "RCP4.5"
    assert _ssp_label("nonexistent") == "nonexistent"
    assert _ssp_color("nonexistent").startswith("#")


def test_pretty_and_short_names_cover_the_same_keys():
    assert set(VARIANT_PRETTY) == set(VARIANT_SHORT)
    assert set(PRODUCT_PRETTY) == set(PRODUCT_SHORT)


def test_pair_labels_default_arms_to_the_titles():
    lab = _pair_labels("A", "B")
    assert (lab["arm_a"], lab["arm_b"]) == ("A", "B")


def test_pair_labels_allow_short_arms_independent_of_long_titles():
    """The in-panel legend sits over the boxes, so it needs short arm names even
    when the column titles are long two-line strings."""
    lab = _pair_labels("A very long column title", "Another long one",
                       arm_a="short A", arm_b="short B")
    assert lab["arm_a"] == "short A"
    assert lab["a"] == "A very long column title"


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------
def test_lighten_moves_toward_white_monotonically():
    base = "#046C9A"
    a, b = _lighten(base, 0.3), _lighten(base, 0.8)
    assert all(0.0 <= c <= 1.0 for c in a + b)
    assert all(hi >= lo for hi, lo in zip(b, a))      # lighter is closer to white
    assert _lighten("#000000", 1.0) == pytest.approx((1.0, 1.0, 1.0))
    assert _lighten("#FFFFFF", 0.5) == pytest.approx((1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# box statistics -- must match what matplotlib draws
# ---------------------------------------------------------------------------
def test_box_stats_reports_quartiles_and_the_mean():
    s = _box_stats(list(range(1, 101)))
    assert s["n_draws"] == 100
    assert s["mean"] == pytest.approx(50.5)
    assert s["median"] == pytest.approx(50.5)
    assert s["q1"] == pytest.approx(25.75)
    assert s["q3"] == pytest.approx(75.25)


def test_whiskers_stop_at_the_last_point_inside_the_fence():
    """A far outlier must sit outside the whisker, and be counted as a flier --
    whisker_hi is what the figure shows, max is the true extreme."""
    values = list(range(1, 101)) + [10_000]
    s = _box_stats(values)
    assert s["max"] == 10_000
    assert s["whisker_hi"] < 200
    assert s["n_fliers"] >= 1


def test_no_fliers_means_whiskers_equal_the_extremes():
    s = _box_stats([10, 11, 12, 13, 14])
    assert s["n_fliers"] == 0
    assert (s["whisker_lo"], s["whisker_hi"]) == (s["min"], s["max"])


def test_interval_is_the_central_95_percent():
    s = _box_stats(list(range(0, 1001)))
    assert s["lower_95"] == pytest.approx(25.0, abs=1.0)
    assert s["upper_95"] == pytest.approx(975.0, abs=1.0)


# ---------------------------------------------------------------------------
# frame reshaping
# ---------------------------------------------------------------------------
def _draws(scale=1.0, n_draws=4) -> pd.DataFrame:
    rows = []
    for draw in range(n_draws):
        for i, year in enumerate((2024, 2025)):
            rows.append(dict(ssp_scenario="ssp245", measure="mortality", year_id=year,
                             draw=draw, count_vacc=10.0 * scale * (i + 1) + draw,
                             count_vacc_cum=10.0 * scale * (i + 1) * (i + 1) + draw))
    return pd.DataFrame(rows)


def test_summarize_generic_emits_mean_and_interval_per_series():
    out = _summarize_generic(_draws(), {"x": "count_vacc"})
    assert {"x_mean", "x_lo", "x_hi"} <= set(out.columns)
    assert len(out) == 2


def test_normalize_vaccine_maps_scenarios_onto_generic_roles():
    summary = pd.DataFrame([dict(ssp_scenario="ssp245", measure="mortality", year_id=2024,
                                 novacc_mean=10.0, novacc_lo=9.0, novacc_hi=11.0,
                                 vacc_mean=8.0, vacc_lo=7.0, vacc_hi=9.0,
                                 averted_mean=2.0, averted_lo=1.0, averted_hi=3.0,
                                 novacc_cum_mean=10.0, novacc_cum_lo=9.0, novacc_cum_hi=11.0,
                                 vacc_cum_mean=8.0, vacc_cum_lo=7.0, vacc_cum_hi=9.0,
                                 averted_cum_mean=2.0, averted_cum_lo=1.0, averted_cum_hi=3.0)])
    draws = pd.DataFrame([dict(ssp_scenario="ssp245", measure="mortality", year_id=2024, draw=0,
                               count_novacc_cum=10.0, count_vacc_cum=8.0, averted_cum=2.0)])
    s, d, through = _normalize_vaccine(summary, draws)
    assert {"a_mean", "b_mean", "d_mean", "a_cum_mean"} <= set(s.columns)
    assert {"a_cum", "b_cum", "d_cum"} <= set(d.columns)
    assert through == 2024


def test_normalize_variants_differences_paired_draws():
    """Draw i of one run is comparable to draw i of the other, so differencing is
    draw-wise; the resulting interval is far tighter than differencing quantiles."""
    a, b = _draws(scale=1.0), _draws(scale=0.5)
    s, d, through = _normalize_variants(a, b)
    assert through == 2025
    row = s[s.year_id == 2024].iloc[0]
    assert row["d_mean"] == pytest.approx(row["a_mean"] - row["b_mean"])
    assert (d.d_cum == d.a_cum - d.b_cum).all()


def test_normalize_variants_refuses_misaligned_runs():
    a = _draws()
    b = _draws()[lambda x: x.draw < 2]          # half the draws
    with pytest.raises(ValueError, match="did not align"):
        _normalize_variants(a, b)
