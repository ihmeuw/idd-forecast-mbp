"""Plotting helpers for the pyGAM modeling sandbox."""

from __future__ import annotations

import matplotlib.pyplot as plt

from idd_forecast_mbp.lib.modeling import specs as specs_mod


def plot_smooths(gam, spec, data, *, width: float = 0.95):
    """Partial-dependence plot of each 1D smooth term in ``spec``.

    One panel per 1D ``s()`` term (``smooth``/``mpi``/``mpd``/``cv``/``cx``);
    linear, factor, and 2D ``Tensor`` terms are skipped (a factor is many country
    levels, a tensor is a 2D surface — neither is a single curve). Curves are on
    the response link scale (logit for PfPR, log for rates) — use them to check
    the fitted *shape* against the intended constraint.

    ``data`` is the frame the model was fit on: pyGAM's ``generate_X_grid`` fills
    held columns with 0, which is out of a factor term's domain, so each factor
    column is pinned to a real level (``data[col].iloc[0]``). The held value
    doesn't affect an isolated term's partial dependence — it only has to be a
    valid level so pyGAM's domain check passes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    layout = specs_mod.spec_layout(spec)
    # (gam_term_index, x-axis flat column, element) for each 1D smooth term
    todo = [(ti, flat[0], e) for (e, ti, flat) in layout
            if not isinstance(e, specs_mod.Tensor) and e.form in specs_mod._SMOOTH_FORMS]
    if not todo:
        raise ValueError("spec has no 1D smooth terms to plot")
    # factor columns need a valid held level (generate_X_grid fills them with 0)
    holds = {flat[0]: data[e.col].iloc[0] for (e, ti, flat) in layout
             if not isinstance(e, specs_mod.Tensor) and e.form == "factor"}

    ncol = min(3, len(todo))
    nrow = -(-len(todo) // ncol)  # ceil division
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for ax, (term_index, xcol, e) in zip(axes.flat, todo):
        grid = gam.generate_X_grid(term=term_index)
        for j, val in holds.items():
            grid[:, j] = val  # pin factor columns to a valid level
        pdep, ci = gam.partial_dependence(term=term_index, X=grid, width=width)
        ax.plot(grid[:, xcol], pdep)
        ax.plot(grid[:, xcol], ci, ls="--", c="grey")
        ax.set_title(f"{e.col} [{e.form}]", fontsize=9)
        ax.set_xlabel(e.col, fontsize=8)
    for ax in axes.flat[len(todo):]:  # blank any unused panels
        ax.axis("off")
    fig.tight_layout()
    return fig
