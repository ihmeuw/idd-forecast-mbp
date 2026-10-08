import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing import scenario_gap as dsg
from idd_forecast_mbp.lib.processing.scenario_gap import (
    FloatArray,
    interp_term,
    log_rate_from_terms,
    logit_suitability,
    shapley,
)


def test_interp_term_is_exact_at_nodes_and_linear_between() -> None:
    xg = np.array([0.0, 1.0, 3.0])
    yg = np.array([0.0, 2.0, 0.0])
    np.testing.assert_allclose(
        interp_term(xg, yg, np.array([0.0, 0.5, 1.0, 2.0, 3.0])), [0, 1, 2, 1, 0]
    )


def test_interp_term_log_grid_interpolates_in_log_x() -> None:
    xg = np.array([1.0, 100.0])
    yg = np.array([0.0, 2.0])
    np.testing.assert_allclose(interp_term(xg, yg, np.array([10.0]), log_x=True), [1.0])


def test_interp_term_raises_outside_the_grid_instead_of_clamping() -> None:
    with pytest.raises(ValueError, match="outside the term grid"):
        interp_term(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.5]))


def test_interp_term_keeps_nan() -> None:
    out = interp_term(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.nan, 0.5])
    )
    assert np.isnan(out[0])
    assert out[1] == 0.5


def test_logit_suitability_clamps_like_the_rocket() -> None:
    out = logit_suitability(np.array([0.0, 182.5, 365.0]))
    lo = np.log(1e-3 / (1 - 1e-3))
    np.testing.assert_allclose(out, [lo, 0.0, -lo])


def test_log_rate_from_terms_is_the_anchor_at_the_rake_year() -> None:
    obs = np.array([-3.0, -5.0])
    lobs = np.array([0.2, -1.0])
    out = log_rate_from_terms(
        obs, lobs, lobs, np.tanh, gdp_slope=-0.05, log_gdp_change=np.zeros(2)
    )
    np.testing.assert_allclose(out, obs)


def test_log_rate_from_terms_adds_the_g_change_and_the_gdp_term() -> None:
    out = log_rate_from_terms(
        np.array([-3.0]),
        np.array([1.0]),
        np.array([0.0]),
        lambda x: 2 * x,
        gdp_slope=-0.5,
        log_gdp_change=np.array([0.4]),
    )
    np.testing.assert_allclose(out, [-3.0 + 2.0 - 0.2])


def test_shapley_of_an_additive_value_is_each_factors_own_effect() -> None:
    effects = {"a": np.array([1.0, 2.0]), "b": np.array([-3.0, 0.5])}
    parts = shapley(lambda s: sum((effects[f] for f in s), np.zeros(2)), ["a", "b"])
    for f, effect in effects.items():
        np.testing.assert_allclose(parts[f], effect)


def test_shapley_splits_a_product_interaction_evenly() -> None:
    # value = x_a * x_b with x_a: 1 -> 3 and x_b: 2 -> 5; gap 13; each part takes half the interaction
    xa = {False: 1.0, True: 3.0}
    xb = {False: 2.0, True: 5.0}
    parts = shapley(lambda s: np.array(xa["a" in s] * xb["b" in s]), ["a", "b"])
    np.testing.assert_allclose(parts["a"], 0.5 * ((3 - 1) * 2 + (3 - 1) * 5))
    np.testing.assert_allclose(parts["b"], 0.5 * ((5 - 2) * 1 + (5 - 2) * 3))


@pytest.mark.parametrize("factors", [["g"], ["g", "s"], ["g", "s", "w"]])
def test_shapley_parts_sum_to_the_full_gap(factors: list[str]) -> None:
    rng = np.random.default_rng(0)
    table: dict[frozenset[str], FloatArray] = {}

    def value(s: frozenset[str]) -> FloatArray:
        if s not in table:
            table[s] = rng.normal(size=4)
        return table[s]

    parts = shapley(value, factors)
    np.testing.assert_allclose(
        sum(parts.values()), value(frozenset(factors)) - value(frozenset())
    )


def test_shapley_rejects_repeated_factors() -> None:
    with pytest.raises(ValueError, match="distinct"):
        shapley(lambda s: np.zeros(1), ["a", "a"])


# ---------------------------------------------------------------------------
# Rebuild, Shapley pairs, roll-up
# ---------------------------------------------------------------------------

# toy model: every term linear so the expected log rate is closed-form
B_DAH, B_GDP, B_SUIT, B_WIND, G_SLOPE, GAMMA = -0.1, -0.3, 0.5, -0.2, 0.8, -0.05


def _grids() -> dict[tuple[str, str], tuple[FloatArray, FloatArray]]:
    gdp_x = np.asarray(np.geomspace(1.0, 1e5, 1000), dtype=np.float64)
    pf_x = np.asarray(np.linspace(-20.0, 20.0, 4001), dtype=np.float64)
    dah_x = np.asarray(np.linspace(0.0, 10.0, 11), dtype=np.float64)
    return {
        ("pfpr", "gdppc_mean"): (gdp_x, B_GDP * np.log(gdp_x)),
        ("pfpr", "mal_DAH_total_per_capita"): (dah_x, B_DAH * dah_x),
        ("inc", "logit_malaria_pfpr"): (pf_x, G_SLOPE * pf_x),
        ("mort", "logit_malaria_pfpr"): (pf_x, G_SLOPE * pf_x),
    }


def _slopes(wind: float = 0.0) -> dict[tuple[str, str], float]:
    return {
        ("pfpr", "logit_malaria_suitability"): B_SUIT,
        ("pfpr", "wind_speed"): wind,
        ("inc", "log_gdppc_mean"): GAMMA,
        ("mort", "log_gdppc_mean"): GAMMA,
    }


def _inputs(gdp_t: float, suit_t: float, wind_t: float) -> dict[str, FloatArray]:
    # 2 locations x 2 years (rake year, t) x 3 draws
    return {
        "gdp": np.array([[100.0, gdp_t], [1000.0, gdp_t * 10]]),
        "dah": np.array([[2.0, 1.0], [0.0, 0.0]]),
        "suit": np.stack([np.full((2, 3), 100.0), np.full((2, 3), suit_t)], axis=1),
        "wind": np.stack([np.full((2, 3), 3.0), np.full((2, 3), wind_t)], axis=1),
    }


def _rebuild(wind: float = 0.0) -> dsg.Rebuild:
    return dsg.Rebuild(
        grids=_grids(),
        slopes=_slopes(wind),
        inputs={"A": _inputs(400.0, 200.0, 4.0), "B": _inputs(200.0, 150.0, 2.0)},
        l_obs=np.array([[-1.0], [0.5]]),
        obs_log={"inc": np.array([[-3.0], [-2.0]]), "mort": np.array([[-7.0], [-6.0]])},
        ry=0,
    )


def _logit_suit(days: FloatArray) -> FloatArray:
    p = days / 365.0
    return np.log(p / (1 - p))


def _expected(
    src_gdp: str, src_suit: str, src_wind: str, wind_slope: float = 0.0
) -> FloatArray:
    rb = _rebuild(wind_slope)
    g, s, w = (rb.inputs[x] for x in (src_gdp, src_suit, src_wind))
    log_g = np.log(g["gdp"][:, 1] / g["gdp"][:, 0])[:, None]
    d_l = (
        B_DAH * (g["dah"][:, 1] - g["dah"][:, 0])[:, None]
        + B_GDP * log_g
        + B_SUIT * (_logit_suit(s["suit"][:, 1]) - _logit_suit(s["suit"][:, 0]))
        + wind_slope * (w["wind"][:, 1] - w["wind"][:, 0])
    )
    return rb.obs_log["inc"] + G_SLOPE * d_l + GAMMA * log_g


def test_align_returns_positions_and_rejects_missing_ids() -> None:
    np.testing.assert_array_equal(
        dsg.align(np.array([5, 7, 9]), np.array([9, 5])), [2, 0]
    )
    with pytest.raises(ValueError, match="missing"):
        dsg.align(np.array([5, 7]), np.array([8]))


def test_rebuild_detects_a_model_with_wind() -> None:
    assert not _rebuild(0.0).wind
    assert _rebuild(B_WIND).wind


@pytest.mark.parametrize("wind", [0.0, B_WIND])
def test_pure_scenario_log_rate_matches_the_closed_form(wind: float) -> None:
    rb = _rebuild(wind)
    parts = rb.year_parts(1)
    for s in ("A", "B"):
        out = rb.log_rate(parts, "inc", dict.fromkeys(dsg.INPUT_FACTORS, s))
        np.testing.assert_allclose(out, _expected(s, s, s, wind), atol=1e-12)


def test_mixed_scenario_takes_each_input_from_its_own_source() -> None:
    rb = _rebuild(B_WIND)
    parts = rb.year_parts(1)
    src = {"gdp": "A", "suitability": "B", "wind": "A"}
    np.testing.assert_allclose(
        rb.log_rate(parts, "inc", src), _expected("A", "B", "A", B_WIND), atol=1e-12
    )


def test_pair_parts_sum_to_the_count_gap_and_respect_the_mask() -> None:
    rb = _rebuild(B_WIND)
    parts = rb.year_parts(1)
    pop = np.array([1e5, 2e5])
    mask = np.array([[False] * 3, [True] * 3])
    out = dsg.pair_parts(
        rb,
        parts,
        measure="inc",
        a="A",
        b="B",
        population=pop,
        mask=mask,
        factors=["gdp", "suitability", "wind"],
    )
    total = out["part_gdp"] + out["part_suitability"] + out["part_wind"]
    np.testing.assert_allclose(total[0], out["count_a"][0] - out["count_b"][0])
    expected_a = (np.exp(_expected("A", "A", "A", B_WIND)) * pop[:, None]).mean(axis=1)
    np.testing.assert_allclose(out["count_a"][0], expected_a[0])
    assert np.isnan(out["count_a"][1])
    assert np.isnan(out["part_gdp"][1])


def test_check_rebuild_is_zero_when_the_forecast_is_the_rebuild() -> None:
    rb = _rebuild()
    parts = rb.year_parts(1)
    forecasts = {
        s: {
            m: np.stack(
                [
                    np.zeros((2, 3)),
                    rb.log_rate(parts, m, dict.fromkeys(dsg.INPUT_FACTORS, s)),
                ],
                axis=1,
            )
            for m in dsg.LOG_RATE_VARS
        }
        for s in ("A", "B")
    }
    checks = dsg.check_rebuild(rb, parts, forecasts, 1)
    assert len(checks) == 4
    assert max(c["max_abs_log_diff"] for c in checks) == 0.0


def _admin2() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair": "A:B",
            "measure": "inc",
            "location_id": [11, 12, 21],
            "year_id": 2050,
            "count_a": [1.0, 2.0, np.nan],
            "part_gdp": [0.5, 0.25, np.nan],
        }
    )


def _keys() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": [11, 12, 21],
            "A0_location_id": [1001, 1001, 1002],
            "region_id": [500, 500, 500],
            "super_region_id": [90, 90, 90],
        }
    )


def test_roll_up_sums_to_each_level_and_keeps_all_nan_groups_nan() -> None:
    out = dsg.roll_up(_admin2(), _keys(), ["count_a", "part_gdp"]).set_index(
        ["level", "location_id"]
    )
    assert out.loc[(3, 1001), "count_a"] == 3.0
    assert np.isnan(out.loc[(3, 1002), "count_a"])
    assert out.loc[(2, 500), "part_gdp"] == 0.75
    assert out.loc[(1, 90), "count_a"] == 3.0
    assert out.loc[(0, 1), "count_a"] == 3.0


def test_roll_up_rejects_admin2_without_a_parent() -> None:
    keys = _keys()
    keys.loc[2, "region_id"] = np.nan
    with pytest.raises(ValueError, match="without a parent"):
        dsg.roll_up(_admin2(), keys, ["count_a"])


def test_roll_up_groups_by_the_given_keys() -> None:
    admin2 = _admin2().drop(columns="pair").assign(arm="ssp245:gdp_held")
    out = dsg.roll_up(admin2, _keys(), ["count_a"], by=("arm", "measure", "year_id"))
    assert set(out.arm) == {"ssp245:gdp_held"}
    assert out.set_index(["level", "location_id"]).loc[(0, 1), "count_a"] == 3.0
