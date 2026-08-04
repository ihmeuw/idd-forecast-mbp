"""Model-summary helpers: per-term p-values and parametric effect coefficients.

pyGAM reports one p-value per *term* — the country factor's is a single joint
p-value across all levels, not one per country (see :func:`term_pvalues`).
:func:`coefficient_table` pulls the interpretable *parametric* coefficients
(linear covariates + per-country factor levels) with standard errors and an
approximate two-sided normal p-value, labelling each country by
``A0_location_id``. Both return DataFrames (``.to_parquet(...)`` to save).

Caveat pyGAM itself prints: with smoothing parameters estimated, these p-values
run smaller than they should — treat them as rough, not exact inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from idd_forecast_mbp.lib.modeling.specs import Tensor


def _term_starts(gam) -> np.ndarray:
    """Coefficient offset of each term in ``gam.coef_`` (len = n_terms + 1)."""
    return np.cumsum([0] + [t.n_coefs for t in gam.terms])


def term_pvalues(gam, spec) -> pd.DataFrame:
    """One row per term (+ intercept): term, form, edf, p_value.

    The factor term's ``p_value`` is the joint significance of the whole country
    effect; smooth terms get their term-level p-value. Aligned to ``gam.terms``
    (spec order, then the intercept).
    """
    starts = _term_starts(gam)
    edof_per_coef = np.asarray(gam.statistics_["edof_per_coef"], dtype=float)
    edf = [float(edof_per_coef[starts[i]:starts[i + 1]].sum()) for i in range(len(gam.terms))]
    labels, forms = [], []
    for e in spec:
        if isinstance(e, Tensor):
            labels.append(f"te({e.cols[0]},{e.cols[1]})")
            forms.append("tensor")
        else:
            labels.append(e.col)
            forms.append(e.form)
    labels.append("intercept")
    forms.append("intercept")
    return pd.DataFrame({"term": labels, "form": forms, "edf": edf,
                         "p_value": list(gam.statistics_["p_values"])})


def _code_to_location(data: pd.DataFrame) -> np.ndarray:
    """A0_location_id indexed by dense code (0..K-1), from the fitted frame."""
    m = data[["A0_af", "A0_location_id"]].drop_duplicates().sort_values("A0_af")
    return m["A0_location_id"].to_numpy()


def coefficient_table(gam, spec, data: pd.DataFrame) -> pd.DataFrame:
    """One row per *parametric* coefficient: linear terms + per-country factor levels.

    Columns: term, form, level (A0_location_id for factor rows, else NaN), coef,
    se, z, p_approx (two-sided normal). Smooth-term basis coefficients are omitted
    (not interpretable effects — use :func:`term_pvalues` for their significance).
    The spec must include at least one linear or factor term.

    Factor level labelling: with one-hot coding pyGAM keeps all K levels; with
    dummy coding it drops the first (lowest-id) level as the reference — the kept
    coefficients are aligned to the last ``n`` country ids accordingly.
    """
    coef = np.asarray(gam.coef_, dtype=float)
    se = np.asarray(gam.statistics_["se"], dtype=float)
    starts = _term_starts(gam)
    locations = _code_to_location(data)

    rows = []
    for i, term in enumerate(spec):
        lo, hi = starts[i], starts[i + 1]
        if isinstance(term, Tensor):
            continue  # 2D tensor basis coefs aren't interpretable effects
        if term.form == "linear":
            rows.append({"term": term.col, "form": "linear", "level": np.nan,
                         "coef": coef[lo], "se": se[lo]})
        elif term.form == "factor":
            kept = locations[-(hi - lo):]  # one-hot: all K; dummy: last K-1 (first is reference)
            for loc_id, c, s in zip(kept, coef[lo:hi], se[lo:hi]):
                rows.append({"term": term.col, "form": "factor", "level": int(loc_id),
                             "coef": c, "se": s})
        # smooth terms: basis coefficients are not interpretable effects -> skipped

    out = pd.DataFrame(rows)
    out["z"] = out["coef"] / out["se"]
    out["p_approx"] = 2.0 * norm.sf(np.abs(out["z"].to_numpy()))
    return out
