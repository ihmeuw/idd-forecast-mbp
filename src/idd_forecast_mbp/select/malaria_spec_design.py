"""Malaria PfPR spec design on ``idd_tools.model_selection``.

Replaces ``build_malaria_neighborhood_specs.r``'s formula-string baking with a typed
:class:`~idd_tools.model_selection.ModelSpace`. Each covariate *group* is one term axis: it is
either ``out`` or enters as one ``(variable, form)`` option; the always-in ``A0_af`` country
fixed effect is the backbone (baked into the formula, not an axis). Enumerating the
:class:`ModelUniverse` reproduces the exact 1,620-spec grid, and :func:`formula_text` rebuilds
the exact R formula the scam / gam / lm worker fits (``deparse1`` formatting, incl.
double-quoted ``bs``).

The library gives us for free what the R builder never had: `config_key` identity (no
`task_id.rsplit("_n")` string surgery), a `complexity_order` per axis (so the M4 parsimony
down-set search is structural), and `to_cellset()` for the jobmon hand-off.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from idd_tools.model_selection import Axis, AxisKind, ModelSpace, ModelUniverse

if TYPE_CHECKING:
    from idd_tools.model_selection import Config

RESPONSE = "logit_malaria_pfpr"
BACKBONE = "A0_af"  # always-in country FE; baked into every formula, not a selection axis
K_DEFAULT = 6
OUT = "out"  # the "group is absent" level (the term-axis bottom)

# Each group's options as (variable, form, k); k is None for a bare linear term. Mirrors the
# `groups` x `var_forms` definition in build_malaria_neighborhood_specs.r exactly.
_GROUPS: dict[str, list[tuple[str, str, int | None]]] = {
    "g1_dah": [("mal_DAH_total_per_capita", "mpd", 4)],
    "g2_gdp": [("gdppc_mean", "mpd", 4)],
    "g3_suit": [("malaria_suit", "mpi", 6), ("logit_malaria_suitability", "linear", None)],
    "g4_temp": [
        ("mean_temperature", "linear", None),
        ("mean_temperature", "smooth", 4),
        ("mean_low_temperature", "linear", None),
        ("mean_low_temperature", "smooth", 4),
    ],
    "g5_urban": [
        ("weighted_1km_urban_threshold_300.0_simple_mean", "linear", None),
        ("weighted_1km_urban_threshold_300.0_simple_mean", "mpd", 4),
    ],
    "g6_precip": [("total_precipitation", "linear", None), ("total_precipitation", "mpi", 4)],
    "g7_humid": [("relative_humidity", "mpi", 6), ("logit_relative_humidity", "linear", None)],
}
AXIS_ORDER: tuple[str, ...] = tuple(_GROUPS)


def _token(var: str, form: str) -> str:
    return f"{var}|{form}"


# level token -> (variable, form, k), for the formula builder and the smooth/scam counts
_TERM: dict[str, tuple[str, str, int | None]] = {
    _token(v, f): (v, f, k) for opts in _GROUPS.values() for (v, f, k) in opts
}


def _axis(name: str, opts: list[tuple[str, str, int | None]]) -> Axis:
    levels = [OUT, *(_token(v, f) for v, f, _ in opts)]
    # complexity: out (0) < a linear term (1) < a smooth/scam term (2). Two parsimony routes.
    order = {OUT: 0} | {_token(v, f): (1 if f == "linear" else 2) for v, f, _ in opts}
    return Axis(name, AxisKind.TERM, levels, bottom=OUT, complexity_order=order)


def build_model_space() -> ModelSpace:
    """The typed covariate-selection space: 7 group axes (A0_af is the baked backbone)."""
    return ModelSpace([_axis(name, opts) for name, opts in _GROUPS.items()])


def build_universe() -> ModelUniverse:
    """Enumerate the full factorial (2*2*3*5*3*3*3 = 1,620 configs)."""
    space = build_model_space()
    levels = [space.axis(name).levels for name in AXIS_ORDER]
    configs = [dict(zip(AXIS_ORDER, combo, strict=True)) for combo in product(*levels)]
    return ModelUniverse.build(space, configs)


def _build_term(var: str, form: str, k: int | None) -> str:
    """One RHS term, matching the R ``build_term`` + ``deparse1`` output exactly."""
    if form == "linear":
        return var
    kk = K_DEFAULT if k is None else k
    if form == "smooth":
        return f"s({var}, k = {kk})"
    return f's({var}, k = {kk}, bs = "{form}")'


def formula_text(config: Config) -> str:
    """Rebuild the exact R formula string for ``config`` (terms in group order, A0_af last)."""
    terms = []
    for name in AXIS_ORDER:
        level = config[name]
        if level == OUT:
            continue
        var, form, k = _TERM[level]
        terms.append(_build_term(var, form, k))
    terms.append(BACKBONE)
    return f"{RESPONSE} ~ " + " + ".join(terms)


def n_smooths(config: Config) -> int:
    """Count of non-linear present terms (drives engine dispatch; A0_af is linear)."""
    return sum(1 for name in AXIS_ORDER if config[name] != OUT and _TERM[config[name]][1] != "linear")


def n_scams(config: Config) -> int:
    """Count of shape-constrained (non-linear, non-smooth) present terms."""
    return sum(
        1
        for name in AXIS_ORDER
        if config[name] != OUT and _TERM[config[name]][1] not in ("linear", "smooth")
    )


def n_terms(config: Config) -> int:
    """Count of present covariate terms — every non-``out`` axis, **linear terms included**
    (unlike :func:`n_smooths`). This is the parsimony complexity: dropping *any* term, linear
    or smooth, makes a model simpler. The always-in ``A0_af`` backbone is not an axis, so it
    doesn't count. By construction ``n_scams <= n_smooths <= n_terms``."""
    return sum(1 for name in AXIS_ORDER if config[name] != OUT)
