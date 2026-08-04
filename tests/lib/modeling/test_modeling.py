"""Unit tests for the pyGAM modeling sandbox (lib/modeling).

Uses small synthetic frames (no parquet, no /tmp) so the suite is fast and
self-contained. Fits are tiny (few terms, low k) — these test the plumbing
(row filters, spec enumeration, fold structure, metric formulas, the 2023
shift, and the downstream chain), not statistical accuracy.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: no display needed for the plot test

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from idd_forecast_mbp.lib.modeling import data as dmod
from idd_forecast_mbp.lib.modeling import fit as fmod
from idd_forecast_mbp.lib.modeling import metrics as mmod
from idd_forecast_mbp.lib.modeling import shift as shmod
from idd_forecast_mbp.lib.modeling import specs as smod
from idd_forecast_mbp.lib.modeling import summary as sumod
from idd_forecast_mbp.lib.modeling import viz as vmod


def _raw_frame(seed: int = 0) -> pd.DataFrame:
    """A synthetic parquet-shaped frame: 6 countries x 3 locs x 24 years (2000-2023)."""
    rng = np.random.default_rng(seed)
    recs = [(c * 100 + j, y, c)
            for c in (10, 20, 30, 40, 50, 60)
            for j in range(3)
            for y in range(2000, 2024)]
    df = pd.DataFrame(recs, columns=["location_id", "year_id", "A0_location_id"])
    df["A0_location_id"] = df["A0_location_id"].astype("int32")
    n = len(df)
    df["population"] = rng.uniform(5e4, 5e5, n)
    df["malaria_pfpr"] = rng.uniform(0.01, 0.6, n)
    df["malaria_inc_rate"] = rng.uniform(0.01, 0.5, n)
    df["malaria_mort_rate"] = rng.uniform(1e-4, 5e-3, n)
    df["logit_malaria_pfpr"] = np.log(df["malaria_pfpr"] / (1 - df["malaria_pfpr"]))
    df["gdppc_mean"] = rng.uniform(500, 20000, n).astype("float32")
    df["mal_DAH_total_per_capita"] = rng.uniform(0.1, 10, n)
    df["ldipc_mean"] = rng.uniform(500, 20000, n).astype("float32")
    df["med_consumppc"] = rng.uniform(500, 5000, n).astype("float32")
    df["days_over_30C"] = rng.uniform(0, 300, n).astype("float32")
    df["relative_humidity"] = rng.uniform(10, 95, n).astype("float32")
    for extra in ("mean_temperature", "mean_low_temperature", "total_precipitation",
                  "weighted_1km_urban_threshold_300.0_simple_mean"):
        df[extra] = rng.uniform(0, 30, n).astype("float32")
    df["malaria_suitability_mordecai_0_0"] = rng.uniform(0, 365, n).astype("float32")
    return df


@pytest.fixture(scope="module")
def modeled_df() -> pd.DataFrame:
    """Cleaned 'final' modeling frame (carries log-rate responses for the chain)."""
    return dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "final")


def _pfpr_spec():
    return (smod.Term("gdppc_mean", "linear"), smod.Term("A0_af", "factor"))


def _downstream_spec():
    # unconstrained low-k smooth keeps the tiny-data fit fast + stable
    return (smod.Term("logit_malaria_pfpr", "smooth", 5), smod.Term("A0_af", "factor"))


# --- data ---------------------------------------------------------------------

def test_apply_transforms_adds_expected_columns():
    df = dmod.apply_transforms(_raw_frame())
    for col in ("malaria_suit", "logit_malaria_suitability", "logit_relative_humidity",
                "logit_do30", "log_gdppc_mean", "malaria_inc_count"):
        assert col in df.columns
    assert np.isfinite(df["logit_malaria_suitability"].to_numpy()).all()  # clip keeps it finite


def test_row_filter_encodes_dense_country_codes():
    # pyGAM's f() needs contiguous 0..K-1 codes; raw location_ids (with gaps) under-fit it.
    df = dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "modeled")
    codes = set(df["A0_af"].unique())
    assert codes == set(range(len(codes)))     # contiguous 0..K-1
    assert "A0_location_id" in df.columns      # raw ids retained for mapping codes -> countries


def test_add_a0_lag():
    df = dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "modeled")
    out = dmod.add_a0_lag(df, "malaria_pfpr", lag_years=10)
    assert "lag10_a0_pfpr" in out.columns
    # earliest years (< min_year + lag) have no lag; later years do
    assert out.loc[out["year_id"] < 2010, "lag10_a0_pfpr"].isna().all()
    assert out.loc[out["year_id"] >= 2010, "lag10_a0_pfpr"].notna().any()
    # value: lag at (country, 2010) == pop-weighted A0 pfpr at (country, 2000)
    c = 10
    s0 = df[(df["A0_location_id"] == c) & (df["year_id"] == 2000)]
    expected = (s0["malaria_pfpr"] * s0["population"]).sum() / s0["population"].sum()
    got = out.loc[(out["A0_location_id"] == c) & (out["year_id"] == 2010), "lag10_a0_pfpr"].iloc[0]
    assert np.isclose(got, expected)
    # logit transform (prevalence)
    outl = dmod.add_a0_lag(df, "malaria_pfpr", lag_years=10, transform="logit")
    assert "logit_lag10_a0_pfpr" in outl.columns
    # default var is pfpr; unknown transform rejected
    assert "lag10_a0_pfpr" in dmod.add_a0_lag(df, lag_years=10).columns
    with pytest.raises(ValueError):
        dmod.add_a0_lag(df, "malaria_pfpr", transform="sqrt")


def test_add_a0_lag_mort_rate_log():
    df = dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "modeled")
    out = dmod.add_a0_lag(df, "malaria_mort_rate", lag_years=10, transform="log")
    assert "lag10_a0_mort_rate" in out.columns
    assert "log_lag10_a0_mort_rate" in out.columns
    # log column is the plain log of the raw lag where both exist
    m = out["lag10_a0_mort_rate"].notna()
    assert np.allclose(out.loc[m, "log_lag10_a0_mort_rate"],
                       np.log(out.loc[m, "lag10_a0_mort_rate"]))


def test_add_a0_lag_is_nan_safe():
    # a partly-missing var: NaN rows must drop out of BOTH numerator and denominator
    df = dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "modeled")
    df = df.copy()
    c = 10
    # blank ONE loc's mort_rate in the source year; the country-year average must ignore
    # that row entirely (its population must NOT inflate the denominator) but still average
    # the surviving locs of the same country-year
    one_loc = df.loc[df["A0_location_id"] == c, "location_id"].iloc[0]
    hole = (df["location_id"] == one_loc) & (df["year_id"] == 2000)
    assert hole.sum() == 1
    df.loc[hole, "malaria_mort_rate"] = np.nan
    out = dmod.add_a0_lag(df, "malaria_mort_rate", lag_years=10)
    kept = df[(df["A0_location_id"] == c) & (df["year_id"] == 2000) & df["malaria_mort_rate"].notna()]
    assert 0 < len(kept) < 3  # some dropped, some survive
    expected = (kept["malaria_mort_rate"] * kept["population"]).sum() / kept["population"].sum()
    got = out.loc[(out["A0_location_id"] == c) & (out["year_id"] == 2010), "lag10_a0_mort_rate"].iloc[0]
    assert np.isclose(got, expected)


def test_add_a0_pfpr_reference():
    df = dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "modeled")
    out = dmod.add_a0_pfpr_reference(df, ref_year=2023)
    assert "a0_pfpr_2023" in out.columns
    # constant within a country across years (broadcast)
    assert (out.groupby("A0_location_id")["a0_pfpr_2023"].nunique(dropna=True) <= 1).all()
    # pop-weighted value
    c = 10
    s = df[(df["A0_location_id"] == c) & (df["year_id"] == 2023)]
    expected = (s["malaria_pfpr"] * s["population"]).sum() / s["population"].sum()
    got = out.loc[out["A0_location_id"] == c, "a0_pfpr_2023"].dropna().iloc[0]
    assert np.isclose(got, expected)
    # simple-mean + logit branches
    om = dmod.add_a0_pfpr_reference(df, ref_year=2023, population_weighted=False, logit=True)
    assert "logit_a0_pfpr_2023" in om.columns
    got_sm = om.loc[om["A0_location_id"] == c, "a0_pfpr_2023"].dropna().iloc[0]
    assert np.isclose(got_sm, s["malaria_pfpr"].mean())


def test_apply_transforms_missing_variant_raises():
    with pytest.raises(KeyError):
        dmod.apply_transforms(_raw_frame(), suit_variant="does_not_exist")


def test_row_filter_modeled():
    df = dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "modeled")
    assert (df["malaria_pfpr"] >= 1e-4).all()
    assert (df["malaria_inc_count"] >= 1).all()


def test_row_filter_final_adds_log_rates():
    df = dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "final")
    assert {"log_malaria_inc_rate", "log_malaria_mort_rate"}.issubset(df.columns)


def test_row_filter_invalid():
    with pytest.raises(ValueError):
        dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "bogus")


# --- specs --------------------------------------------------------------------

def test_generate_specs_defaults_respect_required_and_max_smooths():
    specs = smod.generate_specs()
    assert len(specs) > 0
    for sp in specs:
        forms = {t.col: t.form for t in sp}
        assert forms.get("gdppc_mean") == "mpd"                 # required
        assert forms.get("mal_DAH_total_per_capita") == "mpd"   # required
        assert forms.get("A0_af") == "factor"                   # always-in group g8
        assert smod.n_smooths(sp) <= smod.MAX_SMOOTHS


def test_generate_specs_can_drop_fixed_effects():
    groups = {k: v for k, v in smod.GROUPS.items() if k != "g8"}
    specs = smod.generate_specs(groups=groups)
    assert specs
    assert all("A0_af" not in smod.spec_columns(s) for s in specs)


def test_spec_helpers():
    sp = (smod.Term("gdppc_mean", "mpd", 4), smod.Term("log_gdppc_mean", "linear"),
          smod.Term("A0_af", "factor"))
    assert smod.spec_columns(sp) == ["gdppc_mean", "log_gdppc_mean", "A0_af"]
    assert smod.n_smooths(sp) == 1
    assert smod.spec_to_terms(sp) is not None
    label = smod.spec_label(sp)
    assert "s(gdppc_mean, k=4, mpd)" in label and "factor(A0_af)" in label


# --- metrics ------------------------------------------------------------------

def test_residual_metrics_perfect_fit():
    m = mmod.residual_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["n_obs"] == 3
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r_sq"] == pytest.approx(1.0)
    assert m["r"] == pytest.approx(1.0)


def test_residual_metrics_drops_nonfinite_pairs():
    m = mmod.residual_metrics([1.0, 2.0, np.nan, 4.0], [1.0, 2.0, 3.0, np.inf])
    assert m["n_obs"] == 2


def test_score_predictions_and_by_fold():
    pf = pd.DataFrame({
        "fold": [0, 0, 1, 1],
        "obs": [0.1, 0.2, 0.3, 0.4], "pred": [0.1, 0.25, 0.28, 0.42],
        "obs_natural": [0.5, 0.55, 0.6, 0.65], "pred_natural": [0.51, 0.54, 0.62, 0.64],
    })
    s = mmod.score_predictions(pf)
    assert {"rmse", "r_sq", "r", "nat_r", "nat_rmse"}.issubset(s)
    bf = mmod.score_by_fold(pf)
    assert set(bf["fold"]) == {0, 1, "ALL"}


# --- fit ----------------------------------------------------------------------

def test_fit_predict_in_sample(modeled_df):
    res = fmod.fit_predict(modeled_df, _pfpr_spec(), fmod.PFPR, "IS")
    assert len(res.models) == 1
    p = res.predictions
    assert {"location_id", "year_id", "fold", "obs", "pred",
            "obs_natural", "pred_natural"}.issubset(p.columns)
    assert len(p) == len(modeled_df)                      # IS predicts every row
    assert p["pred_natural"].between(0, 1).all()          # logit link -> (0, 1)


def test_fit_predict_within_pools_all_rows(modeled_df):
    res = fmod.fit_predict(modeled_df, _pfpr_spec(), fmod.PFPR, "within", n_folds=5)
    assert len(res.models) == 5
    assert len(res.predictions) == len(modeled_df)        # each row held out exactly once


def test_fit_predict_temporal_only_test_years(modeled_df):
    res = fmod.fit_predict(modeled_df, _pfpr_spec(), fmod.PFPR, fmod.TEMP_A)
    assert len(res.models) == 1
    assert res.predictions["year_id"].between(2020, 2023).all()


def test_within_country_folds_are_in_range(modeled_df):
    folds = fmod.within_country_folds(modeled_df, 5, 42)
    assert len(folds) == len(modeled_df)
    assert set(np.unique(folds)).issubset(set(range(5)))


def test_fit_predict_drops_nonfinite_response(modeled_df):
    df = modeled_df.copy()
    df.loc[df.index[:10], "logit_malaria_pfpr"] = np.inf
    res = fmod.fit_predict(df, _pfpr_spec(), fmod.PFPR, "IS")
    assert res.n_dropped == 10


# --- shift + chain ------------------------------------------------------------

def test_apply_shift_hits_anchor():
    pf = pd.DataFrame({"location_id": [1, 1, 2, 2], "year_id": [2022, 2023, 2022, 2023],
                       "pred": [0.0, 1.0, 0.0, 2.0]})
    shifted = shmod.apply_shift(pf, {1: 5.0, 2: 5.0}, anchor_year=2023)
    # loc1 shift = 5-1 = 4 -> [4,5]; loc2 shift = 5-2 = 3 -> [3,5]
    assert list(shifted) == [4.0, 5.0, 3.0, 5.0]


def test_anchor_from_observed_takes_finite_anchor_year_rows():
    df = pd.DataFrame({"location_id": [1, 1, 2], "year_id": [2023, 2022, 2023],
                       "resp": [0.5, 0.1, np.nan]})
    assert shmod.anchor_from_observed(df, "resp", 2023) == {1: 0.5}


def test_downstream_chain_temporal(modeled_df):
    res = shmod.downstream_chain(
        modeled_df, _pfpr_spec(), fmod.TEMP_A,
        inc_spec=_downstream_spec(), mort_spec=_downstream_spec(),
    )
    for k in ("pfpr", "inc", "mort"):
        assert res[k]["pred"].shape[0] > 0
        assert "nat_r" in res[k]["score"]
    # pfpr is always shifted -> pred_raw preserved
    assert "pred_raw" in res["pfpr"]["pred"].columns
    tbl = shmod.chain_score_table(res)
    assert list(tbl["outcome"]) == ["pfpr", "inc", "mort"]


def test_downstream_chain_no_downstream_shift(modeled_df):
    res = shmod.downstream_chain(
        modeled_df, _pfpr_spec(), fmod.TEMP_A,
        inc_spec=_downstream_spec(), mort_spec=_downstream_spec(),
        shift_downstream=False,
    )
    assert res["shift_downstream"] is False
    assert "pred_raw" not in res["inc"]["pred"].columns   # inc not shifted


def test_downstream_chain_rejects_within(modeled_df):
    with pytest.raises(NotImplementedError):
        shmod.downstream_chain(modeled_df, _pfpr_spec(), "within",
                               inc_spec=_downstream_spec(), mort_spec=_downstream_spec())


def test_downstream_chain_in_sample(modeled_df):
    res = shmod.downstream_chain(modeled_df, _pfpr_spec(), "IS",
                                 inc_spec=_downstream_spec(), mort_spec=_downstream_spec())
    assert res["evaluation"] == "IS"
    for k in ("pfpr", "inc", "mort"):
        assert res[k]["pred"].shape[0] > 0


# --- coverage of the remaining branches --------------------------------------

def test_default_parquet_path_and_load_pipeline(monkeypatch):
    # load_modeling_data's I/O without touching disk: patch read_parquet to
    # return a synthetic raw frame; path=None also exercises default_parquet_path.
    assert dmod.default_parquet_path().name == "malaria_past_inputs.parquet"
    monkeypatch.setattr(dmod.pd, "read_parquet", lambda *a, **k: _raw_frame())
    out = dmod.load_modeling_data(row_filter="final")
    assert {"A0_af", "log_malaria_inc_rate", "logit_malaria_suitability"}.issubset(out.columns)


def test_row_filter_none_keeps_base_na_drop():
    df = dmod.apply_row_filter(dmod.apply_transforms(_raw_frame()), "none")
    assert "log_malaria_inc_rate" not in df.columns   # only 'final' adds log rates
    assert df["malaria_pfpr"].notna().all()


def test_fit_gam_and_predict_gam(modeled_df):
    gam = fmod.fit_gam(modeled_df, _pfpr_spec(), fmod.PFPR)
    pred = fmod.predict_gam(gam, modeled_df, _pfpr_spec(), fmod.PFPR)
    assert len(pred) == len(modeled_df)
    empty = fmod.predict_gam(gam, modeled_df.iloc[0:0], _pfpr_spec(), fmod.PFPR)
    assert len(empty) == 0


def test_fit_predict_gridsearch(modeled_df):
    res = fmod.fit_predict(modeled_df, _pfpr_spec(), fmod.PFPR, "IS",
                           gridsearch=True, lam_grid=[0.1, 1.0])
    assert len(res.predictions) == len(modeled_df)


def test_inverse_unknown_link_raises():
    with pytest.raises(ValueError):
        fmod._inverse("bogus", np.array([0.0, 1.0]))


def test_fit_predict_unknown_evaluation_raises(modeled_df):
    with pytest.raises(ValueError):
        fmod.fit_predict(modeled_df, _pfpr_spec(), fmod.PFPR, "bogus")


def test_model_stats(modeled_df):
    gam = fmod.fit_gam(modeled_df, _pfpr_spec(), fmod.PFPR)
    stats = mmod.model_stats(gam)
    assert "aic" in stats and "edf" in stats


def test_residual_metrics_empty_with_prefix():
    m = mmod.residual_metrics([np.nan], [np.nan], prefix="nat_")
    assert m["nat_n_obs"] == 0 and np.isnan(m["nat_rmse"])


def test_generate_specs_required_can_filter_and_max_smooths_bounds():
    # a required form that isn't always chosen -> exercises the 'required' skip
    only_suit_mpi = smod.generate_specs(required={"malaria_suit": "mpi"})
    assert all({t.col: t.form for t in s}.get("malaria_suit") == "mpi" for s in only_suit_mpi)
    # default required needs 2 mpd smooths, so max_smooths=1 excludes everything
    assert smod.generate_specs(max_smooths=1) == []


def test_spec_to_terms_unknown_form_raises():
    with pytest.raises(ValueError):
        smod.spec_to_terms((smod.Term("gdppc_mean", "bogus"),))


# --- viz ----------------------------------------------------------------------

def test_plot_smooths(modeled_df):
    # 4 smooths (fills a 2x3 grid -> exercises the blank-unused-panel loop) + a
    # factor that must be skipped
    spec = (smod.Term("gdppc_mean", "smooth", 4),
            smod.Term("mal_DAH_total_per_capita", "smooth", 4),
            smod.Term("mean_temperature", "smooth", 4),
            smod.Term("total_precipitation", "smooth", 4),
            smod.Term("A0_af", "factor"))
    gam = fmod.fit_gam(modeled_df, spec, fmod.PFPR)
    fig = vmod.plot_smooths(gam, spec, modeled_df)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 6            # 2x3 grid: 4 used + 2 blanked
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_smooths_requires_a_smooth():
    with pytest.raises(ValueError):
        vmod.plot_smooths(None, (smod.Term("gdppc_mean", "linear"),
                                 smod.Term("A0_af", "factor")), None)


# --- summary (coefs + p-values) ----------------------------------------------

def _mixed_spec():
    return (smod.Term("gdppc_mean", "linear"),
            smod.Term("logit_malaria_suitability", "smooth", 4),
            smod.Term("A0_af", "factor"))


def test_term_pvalues(modeled_df):
    spec = _mixed_spec()
    gam = fmod.fit_gam(modeled_df, spec, fmod.PFPR)
    tbl = sumod.term_pvalues(gam, spec)
    assert list(tbl["term"]) == ["gdppc_mean", "logit_malaria_suitability", "A0_af", "intercept"]
    assert {"form", "edf", "p_value"}.issubset(tbl.columns)


def test_coefficient_table(modeled_df):
    spec = _mixed_spec()
    gam = fmod.fit_gam(modeled_df, spec, fmod.PFPR)
    tbl = sumod.coefficient_table(gam, spec, modeled_df)
    # 1 linear coef + one row per country level; the smooth is omitted
    n_countries = modeled_df["A0_af"].nunique()
    assert (tbl["form"] == "linear").sum() == 1
    assert (tbl["form"] == "factor").sum() == n_countries          # one-hot keeps all levels
    assert set(tbl.loc[tbl["form"] == "factor", "level"]).issubset(
        set(modeled_df["A0_location_id"].unique()))                # labelled by real country ids
    assert {"coef", "se", "z", "p_approx"}.issubset(tbl.columns)


# --- 2D tensor terms ----------------------------------------------------------

def _tensor_spec():
    return (smod.Tensor(("malaria_suit", "gdppc_mean"), ("mpi", "mpd"), (5, 5)),
            smod.Term("mal_DAH_total_per_capita", "mpd", 5),
            smod.Term("A0_af", "factor"))


def test_tensor_spec_helpers():
    spec = _tensor_spec()
    assert smod.spec_columns(spec) == ["malaria_suit", "gdppc_mean",
                                       "mal_DAH_total_per_capita", "A0_af"]
    assert smod.n_smooths(spec) == 2                       # tensor + DAH smooth
    assert "te(malaria_suit[mpi], gdppc_mean[mpd])" in smod.spec_label(spec)
    layout = smod.spec_layout(spec)
    assert layout[0][2] == (0, 1)                          # tensor spans flat cols 0,1
    assert layout[1][2] == (2,)                            # DAH -> flat col 2
    assert smod.spec_to_terms(spec) is not None


def test_fit_predict_tensor(modeled_df):
    res = fmod.fit_predict(modeled_df, _tensor_spec(), fmod.PFPR, "IS")
    assert len(res.predictions) == len(modeled_df)
    assert res.predictions["pred_natural"].between(0, 1).all()


def test_plot_smooths_skips_tensor(modeled_df):
    gam = fmod.fit_gam(modeled_df, _tensor_spec(), fmod.PFPR)
    fig = vmod.plot_smooths(gam, _tensor_spec(), modeled_df)   # plots only the 1D DAH smooth
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_summary_handles_tensor(modeled_df):
    spec = _tensor_spec()
    gam = fmod.fit_gam(modeled_df, spec, fmod.PFPR)
    pv = sumod.term_pvalues(gam, spec)
    assert pv["term"].astype(str).str.startswith("te(").any()      # tensor labelled
    ct = sumod.coefficient_table(gam, spec, modeled_df)
    assert (ct["form"] == "factor").sum() == modeled_df["A0_af"].nunique()
    assert not ct["term"].astype(str).str.startswith("te(").any()  # tensor coefs skipped
