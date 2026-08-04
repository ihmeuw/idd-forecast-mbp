"""Flexible malaria spec enumerator for pyGAM.

Reimplements the group/form combinatorics of
``03_modeling/build_malaria_neighborhood_specs.r`` as a *flexible* generator
(edit :data:`VAR_FORMS` / :data:`GROUPS`, or pass your own, to explore
formulas), and maps each spec to a pyGAM term structure.

A *spec* is an ordered tuple of :class:`Term` ``(col, form, k)``; the order
defines the columns of the design matrix (pyGAM terms reference features by
integer index). ``form`` is one of:

    linear   -> l(i)                         (bare linear term)
    factor   -> f(i)                         (country fixed effect; penalized in
                                              pyGAM — pass factor_lam~0 to fit_predict
                                              to approximate unpenalized FE)
    smooth   -> s(i, n_splines=k)            (unconstrained thin-plate)
    mpi/mpd  -> s(i, n_splines=k, constraints='monotonic_inc'/'monotonic_dec')
    cv/cx    -> s(i, n_splines=k, constraints='concave'/'convex')

The default menu mirrors the R builder (same covariates, same groups, same
``matches`` filter and MAX_SMOOTHS) with one change: the country term ``A0_af``
is a ``factor`` here (scam treats it as a bare parametric factor).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pygam

# scam bs code -> pyGAM constraint (None = unconstrained smooth)
_CONSTRAINT = {"mpi": "monotonic_inc", "mpd": "monotonic_dec",
               "cv": "concave", "cx": "convex", "smooth": None}
_SMOOTH_FORMS = frozenset({"smooth", "mpi", "mpd", "cv", "cx"})
K_DEFAULT = 6


@dataclass(frozen=True)
class Term:
    col: str
    form: str            # linear | factor | smooth | mpi | mpd | cv | cx
    k: int | None = None  # basis dim for smooth forms; None -> K_DEFAULT


@dataclass(frozen=True)
class Tensor:
    """A 2D tensor-product smooth over two columns (pyGAM ``te()``), with a
    per-marginal shape constraint each, e.g.
    ``Tensor(("malaria_suit", "gdppc_mean"), ("mpi", "mpd"), (6, 6))`` = a 2D
    surface increasing in suitability and decreasing in gdppc. ``forms`` / ``k``
    are ``(dim0, dim1)``; a ``"smooth"`` marginal is unconstrained on that axis.
    Constraints are per-marginal (and penalty-based), not a joint 2D constraint.
    """
    cols: tuple           # (colA, colB)
    forms: tuple          # per-marginal forms, e.g. ("mpi", "mpd")
    k: tuple = (K_DEFAULT, K_DEFAULT)   # n_splines per marginal


# smooth-form -> pyGAM tensor-marginal constraint ("none" = unconstrained margin)
_TENSOR_CONSTRAINT = {"smooth": "none", "mpi": "monotonic_inc", "mpd": "monotonic_dec",
                      "cv": "concave", "cx": "convex"}

Spec = tuple  # tuple[Term | Tensor, ...]


@dataclass(frozen=True)
class Group:
    always_in: bool
    vars: tuple           # tuple[str, ...]


# --- default menu (mirrors build_malaria_neighborhood_specs.r) ----------------
# var -> {form: K}. K is per-(var, form); None for linear/factor. The log_*
# and logit_* alternatives are offered but not wired into the default GROUPS
# (as in the R builder) — they exist for formula experiments.
VAR_FORMS: dict[str, dict[str, int | None]] = {
    "mal_DAH_total_per_capita": {"mpd": 4},
    "log_mal_DAH_total_per_capita": {"linear": None},
    "gdppc_mean": {"mpd": 4},
    "log_gdppc_mean": {"linear": None},
    "malaria_suit": {"mpi": 6},
    "logit_malaria_suitability": {"linear": None},
    "mean_temperature": {"linear": None, "mpi": 4},
    "mean_low_temperature": {"linear": None, "mpi": 4},
    "weighted_1km_urban_threshold_300.0_simple_mean": {"linear": None, "mpd": 4},
    "total_precipitation": {"linear": None, "mpi": 4},
    "relative_humidity": {"mpi": 6},
    "logit_relative_humidity": {"linear": None},
    "A0_af": {"factor": None},
}

GROUPS: dict[str, Group] = {
    "g1": Group(True,  ("mal_DAH_total_per_capita",)),
    "g2": Group(True,  ("gdppc_mean",)),
    "g3": Group(True,  ("malaria_suit", "logit_malaria_suitability")),
    "g4": Group(False, ("mean_temperature", "mean_low_temperature")),
    "g5": Group(False, ("weighted_1km_urban_threshold_300.0_simple_mean",)),
    "g6": Group(False, ("total_precipitation",)),
    "g7": Group(False, ("relative_humidity", "logit_relative_humidity")),
    "g8": Group(True,  ("A0_af",)),
}

# matches(): keep only specs with gdppc=mpd AND DAH=mpd (the R neighborhood filter).
DEFAULT_REQUIRED = {"gdppc_mean": "mpd", "mal_DAH_total_per_capita": "mpd"}
MAX_SMOOTHS = 7


def n_smooths(spec: Spec) -> int:
    """Number of smooth terms (1D s() + 2D te()); linear and factor terms don't count."""
    return sum(isinstance(e, Tensor) or e.form in _SMOOTH_FORMS for e in spec)


def _group_options(group: Group, var_forms: dict) -> list:
    """One option per (var, form) in the group; plus None if the group is optional."""
    opts: list = [] if group.always_in else [None]
    for v in group.vars:
        for form in var_forms[v]:
            opts.append((v, form))
    return opts


def generate_specs(
    groups: dict | None = None,
    var_forms: dict | None = None,
    *,
    required: dict | None = None,
    max_smooths: int | None = MAX_SMOOTHS,
) -> list[Spec]:
    """Cartesian product over group options, filtered by ``required`` + ``max_smooths``.

    Defaults reproduce the R neighborhood. ``required`` maps var -> form and
    keeps only specs where every listed var takes that form (pass ``{}`` to
    disable). Each group contributes at most one var; column order follows group
    order (deterministic), so spec_index is stable for a fixed menu.
    """
    groups = GROUPS if groups is None else groups
    var_forms = VAR_FORMS if var_forms is None else var_forms
    required = DEFAULT_REQUIRED if required is None else required

    per_group = [_group_options(g, var_forms) for g in groups.values()]
    specs: list[Spec] = []
    for combo in product(*per_group):
        chosen: dict[str, str] = {}
        for opt in combo:
            if opt is not None:
                var, form = opt
                chosen[var] = form
        if required and any(chosen.get(v) != f for v, f in required.items()):
            continue
        spec = tuple(Term(v, form, var_forms[v].get(form)) for v, form in chosen.items())
        if max_smooths is not None and n_smooths(spec) > max_smooths:
            continue
        specs.append(spec)
    return specs


def spec_columns(spec: Spec) -> list[str]:
    """Flat design-matrix column names in term order (a Tensor contributes 2)."""
    cols: list[str] = []
    for e in spec:
        if isinstance(e, Tensor):
            cols.extend(e.cols)
        else:
            cols.append(e.col)
    return cols


def spec_layout(spec: Spec) -> list:
    """Per element: ``(element, gam_term_index, flat_column_indices)``.

    Each spec element becomes exactly one gam term (``gam_term_index``); its
    design-matrix column(s) are ``flat_column_indices`` (a Tensor spans two).
    Lets viz/summary map terms <-> columns when a Tensor makes those diverge.
    """
    layout = []
    flat = 0
    for term_index, e in enumerate(spec):
        n = 2 if isinstance(e, Tensor) else 1
        layout.append((e, term_index, tuple(range(flat, flat + n))))
        flat += n
    return layout


def spec_to_terms(spec: Spec, *, factor_lam: float | None = None):
    """Build the pyGAM TermList for a spec (terms reference flat column indices).

    ``factor_lam`` overrides the penalty on factor terms; pass a small value
    (e.g. 1e-3) to approximate unpenalized country fixed effects.
    """
    terms = None
    idx = 0
    for e in spec:
        if isinstance(e, Tensor):
            n_splines = [e.k[0] or K_DEFAULT, e.k[1] or K_DEFAULT]
            constraints = [_TENSOR_CONSTRAINT[f] for f in e.forms]
            term = pygam.te(idx, idx + 1, n_splines=n_splines, constraints=constraints)
            idx += 2
        elif e.form == "linear":
            term = pygam.l(idx)
            idx += 1
        elif e.form == "factor":
            term = pygam.f(idx) if factor_lam is None else pygam.f(idx, lam=factor_lam)
            idx += 1
        elif e.form in _CONSTRAINT:
            k = e.k or K_DEFAULT
            constraint = _CONSTRAINT[e.form]
            term = pygam.s(idx, n_splines=k) if constraint is None else \
                pygam.s(idx, n_splines=k, constraints=constraint)
            idx += 1
        else:
            raise ValueError(f"unknown form {e.form!r} for column {e.col!r}")
        terms = term if terms is None else terms + term
    return terms


def spec_label(spec: Spec) -> str:
    """Human-readable one-line formula (for logging / labelling notebook output)."""
    parts = []
    for e in spec:
        if isinstance(e, Tensor):
            parts.append(f"te({e.cols[0]}[{e.forms[0]}], {e.cols[1]}[{e.forms[1]}])")
        elif e.form == "linear":
            parts.append(e.col)
        elif e.form == "factor":
            parts.append(f"factor({e.col})")
        else:
            parts.append(f"s({e.col}, k={e.k or K_DEFAULT}, {e.form})")
    return " + ".join(parts)
