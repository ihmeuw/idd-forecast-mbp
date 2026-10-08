"""Axes end on a labelled tick: years end at 2100; counts extend to the next labelled value."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from idd_forecast_mbp.lib.viz.axes import (
    end_figure_axes_on_ticks,
    end_on_ticks,
    tick_bounds,
)


def test_year_axis_ends_at_2100_and_counts_at_the_next_tick() -> None:
    fig, ax = plt.subplots()
    ax.plot(np.arange(2000, 2101), np.linspace(512.0, 961.0, 101))
    end_on_ticks(ax, start="at")
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
    end_figure_axes_on_ticks(fig, start="at")
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


def test_default_start_leaves_a_small_gap_before_the_first_label() -> None:
    fig, ax = plt.subplots()
    ax.plot(np.arange(2000, 2101), np.linspace(512.0, 961.0, 101))
    end_on_ticks(ax)
    lo_x, hi_x = ax.get_xlim()
    assert hi_x == 2100.0  # the far end still stops on the last label
    assert lo_x < 2000.0
    assert lo_x == pytest.approx(2000.0 - 0.02 * 100.0)
    assert 2000.0 in ax.get_xticks()
    plt.close(fig)


def test_float_noise_below_a_tick_counts_as_on_it() -> None:
    """Arm-minus-baseline panels are -1e-12 where the runs coincide; the axis must still start at 0."""
    fig, ax = plt.subplots()
    ax.plot([2023, 2050, 2100], [-1e-12, 2500.0, 5000.0])
    end_on_ticks(ax, start="at")
    assert ax.get_ylim() == (0.0, 5000.0)
    plt.close(fig)


def test_ticks_that_do_not_bracket_the_data_are_extended_by_whole_steps() -> None:
    bounds = tick_bounds(-3.0, 997.0)
    assert bounds is not None
    t_lo, t_hi, ticks = bounds
    assert t_lo <= -3.0
    assert t_hi >= 997.0
    assert ticks[0] == t_lo
    assert ticks[-1] == t_hi


def test_a_span_of_pure_float_noise_is_treated_as_flat() -> None:
    """A super-region with no malaria differences to -7e-33..0; shrinking the axis to that blew up a figure."""
    fig, ax = plt.subplots()
    ax.plot([2023, 2050, 2100], [-7.14e-33, 0.0, 0.0])
    before = ax.get_ylim()
    end_on_ticks(ax, x=False)
    assert ax.get_ylim() == before
    assert tick_bounds(-7.14e-33, 0.0) is None
    plt.close(fig)


def test_shared_axes_use_the_union_of_the_group_data() -> None:
    fig, (a, b) = plt.subplots(1, 2, sharey=True)
    a.plot([2023, 2100], [0.0, 940.0])  # the larger values are in the FIRST panel
    b.plot([2023, 2100], [0.0, 410.0])
    end_figure_axes_on_ticks(fig)
    assert a.get_ylim()[1] >= 940.0
    assert b.get_ylim() == a.get_ylim()
    plt.close(fig)
