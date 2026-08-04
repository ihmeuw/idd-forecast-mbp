"""Readable summaries of a fitted pyGAM model.

``LinearGAM.summary()`` prints terms as ``l(0)`` / ``s(6)`` / ``f(9)`` — positional
indices with no column names — and reports no coefficient estimates. That is close
to useless for judging a formulation, so this module rebuilds the summary against
the spec the model was fitted from.

What pyGAM does expose, and this surfaces:

``gam.coef_``
    Every coefficient, concatenated in term order. A linear term contributes one,
    a spline term contributes its basis dimension, a factor term contributes one
    per level.
``gam.statistics_['se']``
    Per-coefficient standard errors, aligned with ``coef_``.
``gam.statistics_['edof_per_coef']``
    Effective degrees of freedom per coefficient.
``gam.statistics_['p_values']``
    One per term, plus the intercept, in term order.

Coefficients are reported for **linear terms only**. A spline term's basis
coefficients are not individually interpretable, and a factor term has one per
level — for those the useful summary is the effective dof and the term p-value,
with the shape read off a partial-dependence plot instead.

pyGAM's p-values are approximate, and it warns that fitting both a spline and a
linear function to the same feature creates an identifiability problem that makes
them unreliable. They are reported because they were asked for; treat them as
indicative, not inferential.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from idd_forecast_mbp.lib.modeling import specs as specs_mod

if TYPE_CHECKING:
    from collections.abc import Sequence

#: pyGAM term class name -> the form we call it.
_TERM_FORM = {
    "LinearTerm": "linear",
    "SplineTerm": "spline",
    "FactorTerm": "factor",
    "TensorTerm": "tensor",
    "Intercept": "intercept",
}


def summarize_gam(gam: Any, spec: Sequence[Any]) -> pd.DataFrame:
    """One row per model term, with the covariate's real name.

    Parameters
    ----------
    gam:
        A fitted pyGAM model.
    spec:
        The spec it was fitted from — used to turn each term's feature index back
        into a column name.

    Returns
    -------
    Columns: ``term`` (covariate name), ``form``, ``n_coef``, ``edf``,
    ``coef`` / ``se`` / ``z`` / ``p_coef`` (linear terms only, else NaN), and
    ``p_term`` (pyGAM's per-term p-value).
    """
    columns = specs_mod.spec_columns(spec)
    stats_ = getattr(gam, "statistics_", {}) or {}
    coefs = np.asarray(gam.coef_, dtype=float)
    se = np.asarray(stats_.get("se", np.full(coefs.shape, np.nan)), dtype=float)
    edof = np.asarray(
        stats_.get("edof_per_coef", np.full(coefs.shape, np.nan)), dtype=float)
    p_terms = np.asarray(stats_.get("p_values", []), dtype=float)

    rows = []
    offset = 0
    for position, term in enumerate(gam.terms):
        n = int(term.n_coefs)
        kind = type(term).__name__
        form = _TERM_FORM.get(kind, kind)
        feature = getattr(term, "feature", None)
        name = ("intercept" if form == "intercept"
                else columns[feature] if feature is not None
                and feature < len(columns) else f"feature_{feature}")

        row: dict[str, Any] = {
            "term": name,
            "form": form,
            "n_coef": n,
            "edf": float(np.nansum(edof[offset:offset + n])),
            "coef": np.nan, "se": np.nan, "z": np.nan, "p_coef": np.nan,
            "p_term": (float(p_terms[position])
                       if position < p_terms.size else np.nan),
        }
        # A single coefficient is interpretable on its own; a basis or a set of
        # factor levels is not.
        if n == 1 and form in ("linear", "intercept"):
            c, s = float(coefs[offset]), float(se[offset])
            row["coef"] = c
            row["se"] = s
            if s > 0:
                row["z"] = c / s
                row["p_coef"] = float(2 * stats.norm.sf(abs(c / s)))
        rows.append(row)
        offset += n

    return pd.DataFrame(rows)


def linear_effects(gam: Any, spec: Sequence[Any]) -> pd.DataFrame:
    """Just the linear terms, as a coefficient table with 95% intervals.

    The subset of :func:`summarize_gam` that reads like a regression table.
    """
    out = summarize_gam(gam, spec)
    out = out[(out["form"] == "linear") & out["coef"].notna()].copy()
    out["ci_low"] = out["coef"] - 1.96 * out["se"]
    out["ci_high"] = out["coef"] + 1.96 * out["se"]
    return out[["term", "coef", "se", "ci_low", "ci_high", "z", "p_coef"]]
