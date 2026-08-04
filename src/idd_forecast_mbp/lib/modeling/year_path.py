"""Effective-year paths: how the time covariate is allowed to evolve in a forecast.

A fitted year term contributes ``beta * year``. After a per-location anchor shift
the forecast carries ``beta * [g(y) - g(anchor)]``, where ``g`` is whatever year
value we *hand the model at predict time*. Because ``beta`` is already estimated,
reshaping ``g`` changes the trajectory **without any refit** — the fitted model is
untouched and only the prediction frame differs.

That is the whole mechanism for decaying a time effect, and it is why the
forecaster must take the year covariate as a supplied per-(location, year) path
rather than computing it from the calendar.

What "the time effect goes to zero" means here
----------------------------------------------
The **marginal** effect goes to zero: each future year contributes less than the
last, until additional years contribute nothing. The cumulative effect plateaus.
It does *not* return to its anchor value — that would undo all accumulated growth
by 2100, which is a different (and much stronger) claim.

So each year past the anchor gets a weight ``w(y)`` in ``[0, 1]``, starting at 1
and reaching 0 at ``end_year``, and

    g(y) = anchor + sum of w(t) for t in (anchor, y]      for y > anchor
    g(y) = y                                              for y <= anchor

Holding the past at identity matters: the fit saw calendar years, so the anchor
window must too, or the shift is computed against a different quantity than the
one that was fitted.

Sizing the effect
-----------------
With no decay the year term runs the full ``2100 - anchor = 77`` years. Under
:func:`linear_decay` the weights average one half, so the effective advance is
~38.5 years — the exponent halves. Since ``beta`` is known (about 0.043 per year
for dengue incidence), the growth factor is ``exp(beta * advance)``: ~27x
undecayed, ~5.3x linear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Year the time effect is required to have stopped growing by.
DEFAULT_END_YEAR = 2100

#: Year the logistic roll-off is centred on by default.
DEFAULT_MIDPOINT_YEAR = 2060


def _path_from_weights(
    years: Sequence[int] | np.ndarray[Any, Any],
    anchor_year: int,
    weight_fn: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
) -> np.ndarray[Any, Any]:
    """Cumulate per-year weights into an effective year, then sample at ``years``.

    The cumulation runs over a DENSE annual grid, not over whatever years the
    caller happened to pass. The effective year at 2050 is the anchor plus every
    annual weight from the anchor to 2050 — asking only about {2023, 2050, 2100}
    must not change it. Cumulating over the passed years directly makes the answer
    depend on the sampling, which is how this went wrong the first time.
    """
    requested = np.asarray(years, dtype=float)
    lo = min(int(anchor_year), int(np.floor(requested.min())))
    hi = max(int(anchor_year), int(np.ceil(requested.max())))
    dense = np.arange(lo, hi + 1, dtype=float)

    weights = np.where(dense > anchor_year, weight_fn(dense), 0.0)
    effective = anchor_year + np.cumsum(weights)
    # The past is untouched: the fit saw calendar years, so must the anchor window.
    effective = np.where(dense <= anchor_year, dense, effective)

    return np.interp(requested, dense, effective)


def identity(years: Sequence[int], anchor_year: int) -> np.ndarray[Any, Any]:
    """No decay: effective year is the calendar year. The default."""
    return np.asarray(years, dtype=float)


def linear_decay(
    years: Sequence[int], anchor_year: int, end_year: int = DEFAULT_END_YEAR,
) -> np.ndarray[Any, Any]:
    """Per-year weight falls linearly from 1 at the anchor to 0 at ``end_year``.

    The simplest way to make the time effect stop: a constant deceleration. The
    effective year advances by about half the calendar span, so the fitted year
    coefficient buys half as much log-change by ``end_year`` as it would undecayed.
    """
    y = np.asarray(years, dtype=float)
    span = float(end_year - anchor_year)
    if span <= 0:
        msg = f"end_year ({end_year}) must be after anchor_year ({anchor_year})"
        raise ValueError(msg)
    def weight(v: np.ndarray[Any, Any]) -> Any:
        return np.clip((end_year - v) / span, 0.0, 1.0)

    return _path_from_weights(y, anchor_year, weight)


def logistic_decay(
    years: Sequence[int],
    anchor_year: int,
    end_year: int = DEFAULT_END_YEAR,
    *,
    midpoint_year: float = DEFAULT_MIDPOINT_YEAR,
    steepness: float = 8.0,
) -> np.ndarray[Any, Any]:
    """Per-year weight follows a logit curve from 1 down to 0.

    Two knobs, which is the point: ``midpoint_year`` is where the roll-off is
    half-spent, and ``steepness`` is how abrupt it is (larger = sharper switch,
    smaller = closer to a gentle ramp). Decay begins immediately at the anchor;
    the midpoint controls where most of it happens.

    The curve is rescaled so the weight is exactly 1 at ``anchor_year`` and
    exactly 0 at ``end_year``, so "stopped by ``end_year``" holds whatever the
    shape parameters are.

    A midpoint at the centre of ``[anchor_year, end_year]`` makes the weights
    symmetric, and symmetric weights average one half — exactly what a linear
    ramp averages. So a centred logistic and :func:`linear_decay` permit the same
    total growth and differ only in when it is spent. Move ``midpoint_year`` off
    centre to change the total: later permits more, earlier permits less.
    """
    y = np.asarray(years, dtype=float)
    span = float(end_year - anchor_year)
    if span <= 0:
        msg = f"end_year ({end_year}) must be after anchor_year ({anchor_year})"
        raise ValueError(msg)
    if not anchor_year < midpoint_year < end_year:
        msg = (f"midpoint_year must lie strictly between anchor_year "
               f"({anchor_year}) and end_year ({end_year}); got {midpoint_year}")
        raise ValueError(msg)

    midpoint = float(midpoint_year)

    def raw(v: np.ndarray[Any, Any] | float) -> Any:
        return 1.0 / (1.0 + np.exp(steepness * (np.asarray(v, float) - midpoint) / span))

    floor_, ceil_ = float(raw(end_year)), float(raw(anchor_year))

    def weight(v: np.ndarray[Any, Any]) -> Any:
        if ceil_ <= floor_:  # pragma: no cover - degenerate steepness
            return np.zeros_like(v)
        scaled = (raw(v) - floor_) / (ceil_ - floor_)
        return np.where(v >= end_year, 0.0, np.clip(scaled, 0.0, 1.0))

    return _path_from_weights(y, anchor_year, weight)


#: Named paths the forecaster and notebook select from.
YEAR_PATHS: dict[str, Callable[..., np.ndarray[Any, Any]]] = {
    "identity": identity,
    "linear_decay": linear_decay,
    "logistic_decay": logistic_decay,
}


def effective_year(
    years: Sequence[int],
    anchor_year: int,
    path: str | Callable[..., np.ndarray[Any, Any]] = "identity",
    **kwargs: Any,
) -> np.ndarray[Any, Any]:
    """Effective year per calendar year, by name or by your own callable.

    Passing a callable is the escape hatch for specifying the math directly: it
    receives ``(years, anchor_year, **kwargs)`` and returns one effective year per
    input year.
    """
    fn = YEAR_PATHS[path] if isinstance(path, str) else path
    return np.asarray(fn(years, anchor_year, **kwargs), dtype=float)


def growth_factor(
    beta: float,
    anchor_year: int,
    target_year: int = DEFAULT_END_YEAR,
    path: str | Callable[..., np.ndarray[Any, Any]] = "identity",
    **kwargs: Any,
) -> float:
    """Multiplicative change a log-link year term produces by ``target_year``.

    ``exp(beta * [g(target) - g(anchor)])``. Lets a decay be judged in the units
    that matter — "how many times higher is incidence in 2100" — before running a
    forecast.
    """
    years = np.arange(anchor_year, target_year + 1)
    g = effective_year(years, anchor_year, path, **kwargs)
    return float(np.exp(beta * (g[-1] - g[0])))
