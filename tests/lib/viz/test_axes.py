"""Axes end on a labelled tick: years end at 2100; counts extend to the next labelled value."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from idd_forecast_mbp.lib.viz.axes import end_figure_axes_on_ticks, end_on_ticks


def test_year_axis_ends_at_2100_and_counts_at_the_next_tick() -> None:
    fig, ax = plt.subplots()
    ax.plot(np.arange(2000, 2101), np.linspace(512.0, 961.0, 101))
    end_on_ticks(ax)
    lo_x, hi_x = ax.get_xlim()
    lo_y, hi_y = ax.get_ylim()
    assert (lo_x, hi_x) == (2000.0, 2100.0)
    assert lo_y <= 512.0 < hi_y
    assert hi_y >= 961.0
    assert hi_y in ax.get_yticks()
    assert lo_y in ax.get_yticks()
    plt.close(fig)


def test_figure_helper_leaves_categorical_x_alone_and_skips_empty_axes() -> None:
    fig, (bars, ts, empty) = plt.subplots(1, 3)
    bars.bar([0, 1, 2], [3.0, 7.3, 5.0])
    bars.set_xlim(-0.7, 2.7)
    ts.plot([2023, 2050, 2100], [1.0, 2.0, 3.0])
    end_figure_axes_on_ticks(fig)
    assert bars.get_xlim() == (-0.7, 2.7)
    assert bars.get_ylim()[1] in bars.get_yticks()
    assert ts.get_xlim()[1] == 2100.0
    assert empty.get_xlim() == (0.0, 1.0)
    plt.close(fig)


def test_flat_or_missing_data_leaves_limits_alone() -> None:
    fig, ax = plt.subplots()
    ax.plot([2000, 2001], [5.0, 5.0])
    before = ax.get_ylim()
    end_on_ticks(ax, x=False)
    assert ax.get_ylim() == before
    plt.close(fig)
