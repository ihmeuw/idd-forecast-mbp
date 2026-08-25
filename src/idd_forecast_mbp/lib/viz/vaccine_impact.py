"""
Presentation helpers for the malaria vaccine impact figures.

Everything here is pure: label building, colour derivation, box statistics and
the frame reshaping that feeds the figures. Figure assembly itself stays in
`08_visualization/plot_vaccine_impact.py`.

Kept separate so the parts that can be checked by assertion are checked by
assertion. NOTE: this deliberately does NOT yet use idd-figures -- porting to
their painters/numbers is a later, explicit step.
"""
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from idd_forecast_mbp import constants as rfc


def _ssp_label(ssp: str) -> str:
    return rfc.ssp_scenario_map.get(ssp, {}).get("name", ssp)


def _ssp_color(ssp: str) -> str:
    return rfc.ssp_scenario_map.get(ssp, {}).get("color", "#B03A2E")


SSP_ORDER = ("ssp126", "ssp245", "ssp585")

MEASURE_LABEL = {"incidence": "malaria cases", "mortality": "malaria deaths"}

NOVACC_COLOR = "#4D4D4D"

LO, HI = 0.025, 0.975

VARIANT_PRETTY = {
    "loglinear_severe0": "severe protection ends at 24 months for non-boosted",
    "loglinear_severeSmooth": "severe protection decays past 24 months for non-boosted",
}

PRODUCT_PRETTY = {
    "projected": "projected rollout (RTS,S + R21)",
    "all_r21": "R21 everywhere",
}

PRODUCT_SHORT = {"projected": "Projected rollout", "all_r21": "R21 everywhere"}

VARIANT_SHORT = {
    "loglinear_severe0": "Severe protection ends at\n24 months for non-boosted",
    "loglinear_severeSmooth": "Severe protection decays past\n24 months for non-boosted",
}

VACCINE_LABELS = {
    "a": "No vaccine", "b": "With vaccine", "d": "Averted",
    "y_level": "Annual {label}", "y_diff": "{label} averted",
    "y_cum": "Cumulative {label}", "y_cum_diff": "Cumulative averted",
    "box": "Cumulative through {through}", "box_diff": "Difference",
    "arm_a": "No vaccine", "arm_b": "With vaccine",
}

def _ascii(text: str) -> str:
    """Fold a display label to ASCII for the stats table.

    Figure titles use a middle dot and em dash, which are fine in a PNG but
    render as mojibake ("Â·") when a spreadsheet opens the CSV as Latin-1. The
    table is meant to be read by eye in Excel, so its labels stay ASCII.
    """
    swaps = {"\u00b7": "|", "\u2014": "-", "\u2013": "-", "\u2022": "|"}
    for bad, good in swaps.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii")

def _cap(text: str) -> str:
    """Sentence-case a label built from a lowercase measure name."""
    return text[:1].upper() + text[1:] if text else text

def _lighten(color: str, amount: float = 0.78):
    """Blend a colour toward white. Used so the no-vaccine box keeps its
    scenario identity instead of going grey."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

def _box_stats(values) -> dict:
    """Mean, 95% interval, and the five-number summary AS DRAWN.

    `whisker_lo/hi` are the most extreme observations within 1.5 x IQR of the
    quartiles -- matplotlib's whisker rule -- so they are the whisker ends in the
    figure, which differ from min/max whenever there are fliers. Both are
    reported so nothing has to be inferred from the picture.
    """
    values = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    return {
        "n_draws": int(values.size),
        "mean": float(values.mean()),
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
        "min": float(values.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(values.max()),
        "whisker_lo": float(inside.min()),
        "whisker_hi": float(inside.max()),
        "n_fliers": int(values.size - inside.size),
    }

def _summarize_generic(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """mean/lo/hi per (ssp, measure, year) for each {out_name: in_col}."""
    agg = {}
    for name, col in cols.items():
        agg[f"{name}_mean"] = (col, "mean")
        agg[f"{name}_lo"] = (col, lambda s: s.quantile(LO))
        agg[f"{name}_hi"] = (col, lambda s: s.quantile(HI))
    return (df.groupby(["ssp_scenario", "measure", "year_id"]).agg(**agg).reset_index())

def _normalize_vaccine(summary: pd.DataFrame, draws: pd.DataFrame):
    """No-vaccine vs with-vaccine, straight from the published summary."""
    ren = {}
    for a, b in (("novacc", "a"), ("vacc", "b"), ("averted", "d"),
                 ("novacc_cum", "a_cum"), ("vacc_cum", "b_cum"), ("averted_cum", "d_cum")):
        for stat in ("mean", "lo", "hi"):
            ren[f"{a}_{stat}"] = f"{b}_{stat}"
    s = summary.rename(columns=ren)
    last = int(draws["year_id"].max())
    d = draws[draws.year_id == last].rename(columns={
        "count_novacc_cum": "a_cum", "count_vacc_cum": "b_cum", "averted_cum": "d_cum"})
    return s, d, last

def _normalize_variants(draws_a: pd.DataFrame, draws_b: pd.DataFrame,
                        key=("ssp_scenario", "measure", "year_id", "draw")):
    """severe0 vs severeSmooth, differenced DRAW-WISE.

    Both runs use the same forecast draws, so draw i of one is comparable to
    draw i of the other. Differencing paired draws and then summarizing gives
    the correct (and much tighter) interval; differencing two sets of published
    quantiles would not be a quantile of the difference at all.

    The series compared are the REMAINING burden under each variant (i.e. each
    one's with-vaccine burden), so the difference is the extra burden that
    severeSmooth prevents relative to severe0.
    """
    cols = list(key) + ["count_vacc", "count_vacc_cum"]
    m = draws_a[cols].merge(draws_b[cols], on=list(key), suffixes=("_a", "_b"))
    if len(m) != len(draws_a):
        raise ValueError(f"variant draws did not align 1:1 ({len(draws_a)} vs {len(m)} merged)")
    m["a"] = m["count_vacc_a"]
    m["b"] = m["count_vacc_b"]
    m["d"] = m["a"] - m["b"]
    m["a_cum"] = m["count_vacc_cum_a"]
    m["b_cum"] = m["count_vacc_cum_b"]
    m["d_cum"] = m["a_cum"] - m["b_cum"]
    s = _summarize_generic(m, {k: k for k in ("a", "b", "d", "a_cum", "b_cum", "d_cum")})
    last = int(m["year_id"].max())
    return s, m[m.year_id == last], last

def _pair_labels(a_title: str, b_title: str,
                 arm_a: str | None = None, arm_b: str | None = None) -> dict:
    """Labels for a figure comparing two runs, `a` as the reference.

    Titles are kept short; the arms are distinguished by a legend rather than a
    parenthetical in the title, which is what used to overlap the neighbouring
    column.
    """
    return {
        "a": a_title, "b": b_title, "d": "Additional averted",
        "y_level": "Annual {label} remaining",
        "y_diff": "Additional {label} averted",
        "y_cum": "Cumulative {label} remaining",
        "y_cum_diff": "Cumulative additional averted",
        "box": f"Cumulative through {{through}}",
        "box_diff": "Difference",
        # Arm labels feed the in-panel legend, which sits over the boxes -- they
        # must stay short even when the column titles are long.
        "arm_a": arm_a or a_title, "arm_b": arm_b or b_title,
    }
