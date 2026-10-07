"""The malaria result arms as data: ``reports/model_runs/malaria_arms.yaml``.

One file names every arm (Baseline plus one change; DECISIONS 2026-10-07) and the forecast
launch each model's arm reads. Finishing, the FHS deliverable and the admin-2 recreation all
resolve an arm through ``load_arms`` so no product can disagree with another about what an arm is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VACCINE_PRODUCTS = ("none", "projected", "all_r21")
_ARM_KEYS = {"forecast", "dah", "vaccine", "population_hold", "age_structure_hold"}
_TOP_KEYS = {"ve_variant", "hold_year", "selection_run", "models", "arms"}


@dataclass(frozen=True)
class Arm:
    """One result arm for one model, resolved to the forecast launch it reads."""

    name: str
    model: str
    forecast_label: str
    dah: str
    vaccine: str
    population_hold_year: int | None
    age_structure_hold_year: int | None

    @property
    def run_key(self) -> str:
        """Product-node name: ``<model>__<arm>``."""
        return f"{self.model}__{self.name}"


@dataclass(frozen=True)
class ArmsConfig:
    ve_variant: str
    hold_year: int
    selection_run: str
    models: dict[str, dict[str, Any]]
    arms: dict[str, dict[str, Any]]
    source: Path

    def arm(self, model: str, name: str) -> Arm:
        """``name`` for ``model``; refuses an arm whose forecast launch the model does not have."""
        if model not in self.models:
            msg = f"unknown model {model!r}; known: {sorted(self.models)}"
            raise KeyError(msg)
        if name not in self.arms:
            msg = f"unknown arm {name!r}; known: {sorted(self.arms)}"
            raise KeyError(msg)
        spec = self.arms[name]
        runs = self.models[model]["forecast_runs"]
        if spec["forecast"] not in runs:
            msg = f"arm {name!r} needs forecast launch {spec['forecast']!r}, which {model} does not have"
            raise KeyError(msg)
        return Arm(
            name=name,
            model=model,
            forecast_label=runs[spec["forecast"]],
            dah=spec["dah"],
            vaccine=spec["vaccine"],
            population_hold_year=self.hold_year
            if spec.get("population_hold")
            else None,
            age_structure_hold_year=self.hold_year
            if spec.get("age_structure_hold")
            else None,
        )

    def arms_for(self, model: str) -> list[Arm]:
        """Every arm the model's forecast launches support, in file order."""
        runs = self.models[model]["forecast_runs"]
        return [
            self.arm(model, n) for n, s in self.arms.items() if s["forecast"] in runs
        ]


def load_arms(path: str | Path) -> ArmsConfig:
    """Read and validate the arms file; unknown keys and bad values refuse."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if set(raw) != _TOP_KEYS:
        msg = f"{path}: top-level keys must be {sorted(_TOP_KEYS)}, got {sorted(raw)}"
        raise ValueError(msg)
    for name, spec in raw["arms"].items():
        extra = set(spec) - _ARM_KEYS
        if extra:
            msg = f"arm {name!r}: unknown key(s) {sorted(extra)}"
            raise ValueError(msg)
        if spec.get("vaccine") not in VACCINE_PRODUCTS:
            msg = f"arm {name!r}: vaccine must be one of {VACCINE_PRODUCTS}"
            raise ValueError(msg)
        if spec.get("population_hold") and spec.get("age_structure_hold"):
            msg = f"arm {name!r}: one change per arm (both holds set)"
            raise ValueError(msg)
    return ArmsConfig(
        ve_variant=str(raw["ve_variant"]),
        hold_year=int(raw["hold_year"]),
        selection_run=str(raw["selection_run"]),
        models=raw["models"],
        arms=raw["arms"],
        source=path,
    )
