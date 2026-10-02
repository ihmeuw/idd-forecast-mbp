"""Malaria PfPR spec design on ``idd_tools.model_selection``.

Replaces ``build_malaria_neighborhood_specs.r``'s formula-string baking with a typed
:class:`~idd_tools.model_selection.ModelSpace`. Each covariate *group* is one term axis: it is
either ``out`` or enters as one ``(variable, form)`` option; the always-in ``A0_af`` country
fixed effect is the backbone (baked into the formula, not an axis). Enumerating the
:class:`ModelUniverse` gives the spec grid, and :meth:`SpecDesign.formula_text` rebuilds the
exact R formula the scam / gam / lm worker fits (``deparse1`` formatting, incl. double-quoted
``bs``).

A design is a property of a selection run, not of the code: the spec-table builder writes the
groups it enumerated into ``<run_dir>/spec_design.json``, and everything that maps a run's
``spec_index`` back to a config (rank, gate, report) reads them from there with
:func:`load_design`. ``_GROUPS`` below is only the design the NEXT run is built from.

The library gives us for free what the R builder never had: `config_key` identity (no
`task_id.rsplit("_n")` string surgery), a `complexity_order` per axis (so the M4 parsimony
down-set search is structural), and `to_cellset()` for the jobmon hand-off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

from idd_tools.model_selection import Axis, AxisKind, ModelSpace, ModelUniverse

if TYPE_CHECKING:
    from idd_tools.model_selection import Config

RESPONSE = "logit_malaria_pfpr"
BACKBONE = (
    "A0_af"  # always-in country FE; baked into every formula, not a selection axis
)
K_DEFAULT = 6
OUT = "out"  # the "group is absent" level (the term-axis bottom)
DESIGN_FILE = "spec_design.json"

Groups = dict[str, list[tuple[str, str, int | None]]]

# The design the next selection run is built from. Each group's options are (variable, form, k);
# k is None for a bare linear term.
_GROUPS: Groups = {
    "g1_dah": [("mal_DAH_total_per_capita", "mpd", 4)],
    "g2_gdp": [("gdppc_mean", "mpd", 4)],
    "g3_suit": [
        ("malaria_suit", "mpi", 6),
        ("logit_malaria_suitability", "linear", None),
    ],
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
    "g6_precip": [
        ("total_precipitation", "linear", None),
        ("total_precipitation", "mpi", 4),
        ("precipitation_days", "linear", None),
        ("precipitation_days", "mpi", 4),
    ],
    "g7_humid": [
        ("relative_humidity", "mpi", 6),
        ("logit_relative_humidity", "linear", None),
    ],
    "g8_wind": [("wind_speed", "linear", None), ("wind_speed", "smooth", 4)],
}


def _token(var: str, form: str) -> str:
    return f"{var}|{form}"


def _build_term(var: str, form: str, k: int | None) -> str:
    """One RHS term, matching the R ``build_term`` + ``deparse1`` output exactly."""
    if form == "linear":
        return var
    kk = K_DEFAULT if k is None else k
    if form == "smooth":
        return f"s({var}, k = {kk})"
    return f's({var}, k = {kk}, bs = "{form}")'


@dataclass(frozen=True)
class SpecDesign:
    """One selection run's covariate groups and everything derived from them."""

    groups: Groups = field(hash=False)

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(self.groups)

    @cached_property
    def _term(self) -> dict[str, tuple[str, str, int | None]]:
        # level token -> (variable, form, k), for the formula builder and the smooth/scam counts
        return {
            _token(v, f): (v, f, k)
            for opts in self.groups.values()
            for (v, f, k) in opts
        }

    def _axis(self, name: str) -> Axis:
        opts = self.groups[name]
        levels = [OUT, *(_token(v, f) for v, f, _ in opts)]
        # complexity: out (0) < a linear term (1) < a smooth/scam term (2). Two parsimony routes.
        order = {OUT: 0} | {
            _token(v, f): (1 if f == "linear" else 2) for v, f, _ in opts
        }
        return Axis(name, AxisKind.TERM, levels, bottom=OUT, complexity_order=order)

    @cached_property
    def space(self) -> ModelSpace:
        """The typed covariate-selection space: one term axis per group (A0_af is the backbone)."""
        return ModelSpace([self._axis(name) for name in self.axes])

    @cached_property
    def universe(self) -> ModelUniverse:
        """The full factorial over the groups; ``configs[i]`` is ``spec_index`` ``i + 1``."""
        levels = [self.space.axis(name).levels for name in self.axes]
        configs = [
            dict(zip(self.axes, combo, strict=True)) for combo in product(*levels)
        ]
        return ModelUniverse.build(self.space, configs)

    def formula_text(self, config: Config) -> str:
        """Rebuild the exact R formula string for ``config`` (terms in group order, A0_af last)."""
        terms = []
        for name in self.axes:
            level = config[name]
            if level == OUT:
                continue
            terms.append(_build_term(*self._term[level]))
        terms.append(BACKBONE)
        return f"{RESPONSE} ~ " + " + ".join(terms)

    def n_smooths(self, config: Config) -> int:
        """Count of non-linear present terms (drives engine dispatch; A0_af is linear)."""
        return sum(
            1
            for name in self.axes
            if config[name] != OUT and self._term[config[name]][1] != "linear"
        )

    def n_scams(self, config: Config) -> int:
        """Count of shape-constrained (non-linear, non-smooth) present terms."""
        return sum(
            1
            for name in self.axes
            if config[name] != OUT
            and self._term[config[name]][1] not in ("linear", "smooth")
        )

    def n_terms(self, config: Config) -> int:
        """Count of present covariate terms — every non-``out`` axis, **linear terms included**
        (unlike :meth:`n_smooths`). This is the parsimony complexity: dropping *any* term, linear
        or smooth, makes a model simpler. The always-in ``A0_af`` backbone is not an axis, so it
        doesn't count. By construction ``n_scams <= n_smooths <= n_terms``."""
        return sum(1 for name in self.axes if config[name] != OUT)

    def to_json(self) -> str:
        return json.dumps(
            {name: [list(opt) for opt in opts] for name, opts in self.groups.items()},
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> SpecDesign:
        raw = json.loads(text)
        return cls(
            {name: [(v, f, k) for v, f, k in opts] for name, opts in raw.items()}
        )


CURRENT_DESIGN = SpecDesign(_GROUPS)


def write_design(design: SpecDesign, run_dir: str | Path) -> Path:
    """Record ``design`` as the run's design; refuses to replace one already there."""
    path = Path(run_dir) / DESIGN_FILE
    if path.exists():
        msg = f"{path} already exists; a run's design is written once"
        raise FileExistsError(msg)
    path.write_text(design.to_json())
    return path


def load_design(run_dir: str | Path) -> SpecDesign:
    """The design a selection run was built from (``<run_dir>/spec_design.json``)."""
    path = Path(run_dir) / DESIGN_FILE
    if not path.is_file():
        msg = f"no {DESIGN_FILE} in {run_dir}; the run predates per-run designs"
        raise FileNotFoundError(msg)
    return SpecDesign.from_json(path.read_text())
