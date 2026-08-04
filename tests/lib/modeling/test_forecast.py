"""Unit tests for the compounded rolling forecast harness (lib/modeling/forecast).

Synthetic frames only (no parquet). The two load-bearing tests:
  - depth-0 == observed-covariate OOS  (plumbing: frame build / predict / aggregate)
  - depth-1 lag == a0(pop-wt mean of depth-0 predictions), logit'd  (the feedback,
    year-shift, aggregation-space, and depth counting — the half that matters)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.modeling import fit as fmod
from idd_forecast_mbp.lib.modeling import forecast as fcmod
from idd_forecast_mbp.lib.modeling import metrics as mmod
from idd_forecast_mbp.lib.modeling import specs as smod

LAG = fcmod.LagCovariate("logit_lag2_a0_pfpr", "malaria_pfpr", 2, "logit")
TRAIN_END = 2010
FC_YEARS = range(2011, 2017)   # 2011-2016: depths 0 (11,12), 1 (13,14), 2 (15,16)


def _frame(seed: int = 0) -> pd.DataFrame:
    """4 countries x 3 locs x years 2000-2016, with a PfPR response + one covariate."""
    rng = np.random.default_rng(seed)
    recs = [(c * 10 + j, y, c)
            for c in (10, 20, 30, 40)
            for j in range(3)
            for y in range(2000, 2017)]
    df = pd.DataFrame(recs, columns=["location_id", "year_id", "A0_location_id"])
    n = len(df)
    df["population"] = rng.uniform(1e4, 1e5, n)
    df["malaria_pfpr"] = rng.uniform(0.02, 0.5, n)
    df["logit_malaria_pfpr"] = np.log(df["malaria_pfpr"] / (1 - df["malaria_pfpr"]))
    df["gdppc_mean"] = rng.uniform(500, 20000, n)
    df["log_gdppc_mean"] = np.log(df["gdppc_mean"])
    df["A0_af"] = df["A0_location_id"].astype("category").cat.codes.astype("int64")
    return df


def _spec():
    return (smod.Term("log_gdppc_mean", "linear"),
            smod.Term("logit_lag2_a0_pfpr", "smooth", 4))


# --- the two correctness gates --------------------------------------------------

def test_depth0_reproduces_observed_oos():
    """At depth 0 the rolled lag is built from observed data, so predictions must
    match the observed-covariate OOS (fit.fit_predict + fit.Temporal)."""
    df = _frame()
    observed, rolled = fcmod.compare_modes(
        df, _spec(), LAG, train_lo=2000, train_end=TRAIN_END, forecast_years=FC_YEARS)

    d0 = rolled.predictions[rolled.predictions["recursion_depth"] == 0]
    assert set(d0["year_id"]) == {2011, 2012}          # the only observed-sourced years for L=2
    m = d0.merge(observed.predictions, on=["location_id", "year_id"], suffixes=("_r", "_o"))
    assert len(m) > 0
    assert np.allclose(m["pred_r"], m["pred_o"], rtol=1e-6, atol=1e-8)
    assert np.allclose(m["pred_natural_r"], m["pred_natural_o"], rtol=1e-6, atol=1e-8)


def test_depth1_lag_is_fed_from_depth0_predictions():
    """The lag at year 2013 (depth 1) must equal logit(pop-wt a0 mean of the
    depth-0 predictions at 2011) — the feedback, +L year shift, aggregation space,
    and transform. A sign flip / off-by-one / pre-vs-post-transform bug fails here
    while passing the depth-0 gate."""
    df = _frame()
    rolled = fcmod.rolling_forecast(
        df, _spec(), LAG, train_end=TRAIN_END, train_lo=2000, forecast_years=FC_YEARS)
    preds = rolled.predictions

    # depth-0 predictions at 2011 -> pop-weighted admin-0 mean per country
    p11 = (preds[preds["year_id"] == 2011][["location_id", "pred_natural"]]
           .merge(df[df["year_id"] == 2011][["location_id", "A0_location_id", "population"]],
                  on="location_id"))
    num = (p11["pred_natural"] * p11["population"]).groupby(p11["A0_location_id"]).sum()
    den = p11["population"].groupby(p11["A0_location_id"]).sum()
    frac = np.clip((num / den).to_numpy(), 1e-3, 1 - 1e-3)
    expected = pd.Series(np.log(frac / (1 - frac)), index=(num / den).index)

    # actual injected lag at 2013 (depth 1), one value per country
    lag13 = (preds[preds["year_id"] == 2013][["location_id", "logit_lag2_a0_pfpr"]]
             .merge(df[["location_id", "A0_location_id"]].drop_duplicates(), on="location_id"))
    actual = lag13.groupby("A0_location_id")["logit_lag2_a0_pfpr"].first()

    assert set(preds[preds["year_id"] == 2013]["recursion_depth"]) == {1}
    for a in expected.index:
        assert np.isclose(actual[a], expected[a], rtol=1e-6), a


# --- stamping, scoring, validation ---------------------------------------------

def test_depth_and_seed_stamping():
    df = _frame()
    preds = fcmod.rolling_forecast(
        df, _spec(), LAG, train_end=TRAIN_END, train_lo=2000, forecast_years=FC_YEARS).predictions
    depth = preds.groupby("year_id")["recursion_depth"].first().to_dict()
    seed = preds.groupby("year_id")["seed_year"].first().to_dict()
    assert depth == {2011: 0, 2012: 0, 2013: 1, 2014: 1, 2015: 2, 2016: 2}
    # seed_year = t - (depth+1)*L  (grounded observed year)
    assert seed == {2011: 2009, 2012: 2010, 2013: 2009, 2014: 2010, 2015: 2009, 2016: 2010}
    assert (preds["is_seeded_from_observed"] == (preds["recursion_depth"] == 0)).all()


def test_score_by_depth_shape():
    df = _frame()
    preds = fcmod.rolling_forecast(
        df, _spec(), LAG, train_end=TRAIN_END, train_lo=2000, forecast_years=FC_YEARS).predictions
    tab = mmod.score_by_depth(preds)
    assert set(tab["recursion_depth"]) >= {0, 1, 2, "ALL"}
    for col in ("n_obs", "nat_n_obs", "nat_r", "nat_rmse", "r_sq", "rmse"):
        assert col in tab.columns
    # per-depth counts sum to the pooled ALL count
    per = tab[tab["recursion_depth"] != "ALL"]["n_obs"].sum()
    allc = tab[tab["recursion_depth"] == "ALL"]["n_obs"].iloc[0]
    assert per == allc


def test_validation_rejects_non_consecutive_and_bad_lags():
    df = _frame()
    with pytest.raises(ValueError):   # step > 1 orphans chains
        fcmod.rolling_forecast(df, _spec(), LAG, train_end=TRAIN_END, forecast_years=range(2011, 2017, 2))
    with pytest.raises(ValueError):   # start != train_end + 1
        fcmod.rolling_forecast(df, _spec(), LAG, train_end=TRAIN_END, forecast_years=range(2012, 2016))
    with pytest.raises(ValueError):   # two rolled lags -> ambiguous depth
        fcmod.rolling_forecast(df, _spec(), [LAG, LAG], train_end=TRAIN_END, forecast_years=range(2011, 2013))
    with pytest.raises(ValueError):   # rolled lag whose source isn't the response
        bad = fcmod.LagCovariate("lag2_a0_gdppc_mean", "gdppc_mean", 2, None, roll=True)
        fcmod.rolling_forecast(df, _spec(), bad, train_end=TRAIN_END, forecast_years=range(2011, 2013))
