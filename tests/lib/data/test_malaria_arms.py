"""The arms file: the committed one loads, every arm resolves, bad files refuse."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from idd_forecast_mbp.lib.data.malaria_arms import load_arms

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parents[3]
ARMS = REPO / "reports" / "model_runs" / "malaria_arms.yaml"


def test_committed_arms_match_the_decision() -> None:
    cfg = load_arms(ARMS)
    assert cfg.ve_variant == "loglinear_severe0"
    assert cfg.hold_year == 2023
    names_7426 = [a.name for a in cfg.arms_for("spec7426_best")]
    names_7427 = [a.name for a in cfg.arms_for("spec7427_alternate")]
    assert names_7426 == [
        "Baseline",
        "DAH_Constant",
        "gdppc_hold2023",
        "population_hold2023",
        "population_age_structure_hold2023",
        "no_vaccine",
        "good_vaccine",
    ]
    assert names_7427 == [*names_7426[:3], "wind_hold2023", *names_7426[3:]]


def test_arm_resolves_its_launch_and_its_one_change() -> None:
    cfg = load_arms(ARMS)
    base = cfg.arm("spec7426_best", "Baseline")
    assert (base.forecast_label, base.dah, base.vaccine) == (
        "spec7426_best__base",
        "Baseline",
        "projected",
    )
    assert base.population_hold_year is None
    assert base.age_structure_hold_year is None
    assert base.run_key == "spec7426_best__Baseline"
    assert cfg.arm("spec7426_best", "population_hold2023").population_hold_year == 2023
    assert (
        cfg.arm(
            "spec7426_best", "population_age_structure_hold2023"
        ).age_structure_hold_year
        == 2023
    )
    assert cfg.arm("spec7426_best", "good_vaccine").vaccine == "all_r21"
    assert cfg.arm("spec7426_best", "no_vaccine").vaccine == "none"
    assert (
        cfg.arm("spec7427_alternate", "wind_hold2023").forecast_label
        == "spec7427_alternate__wind_hold2023"
    )


@pytest.mark.parametrize(
    ("model", "arm", "match"),
    [
        ("nope", "Baseline", "unknown model"),
        ("spec7426_best", "nope", "unknown arm"),
        ("spec7426_best", "wind_hold2023", "does not have"),
    ],
)
def test_bad_lookups_refuse(model: str, arm: str, match: str) -> None:
    with pytest.raises(KeyError, match=match):
        load_arms(ARMS).arm(model, arm)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.pop("ve_variant"), "top-level keys"),
        (lambda r: r["arms"]["Baseline"].update(extra=1), "unknown key"),
        (lambda r: r["arms"]["Baseline"].update(vaccine="maybe"), "vaccine must be"),
        (
            lambda r: r["arms"]["Baseline"].update(
                population_hold=True, age_structure_hold=True
            ),
            "one change per arm",
        ),
    ],
)
def test_bad_files_refuse(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], Any], match: str
) -> None:
    raw = yaml.safe_load(ARMS.read_text())
    mutate(raw)
    p = tmp_path / "arms.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=match):
        load_arms(p)
