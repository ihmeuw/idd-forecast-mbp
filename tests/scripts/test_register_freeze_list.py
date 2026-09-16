"""The freeze-list registration plan is internally consistent and matches the agreed list."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "register_freeze_list.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("register_freeze_list", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = (
        m  # dataclasses resolve postponed annotations through sys.modules
    )
    spec.loader.exec_module(m)
    return m


def test_plan_is_consistent(mod):
    assert mod.check_plan() == []


def test_plan_covers_the_agreed_currents_and_labels(mod):
    rows = mod.plan_rows()
    currents = {(r["node"], r["directory"]) for r in rows if r["current"]}
    assert (
        "03-modeling_data/malaria/scam_prelim/lsae_1285",
        "20260727_efs",
    ) in currents
    assert (
        "04-forecasting_data/malaria/forecast_outputs/lsae_1285",
        "2026_07_31_full_model_selection_results__gdpscen",
    ) in currents
    assert (
        "04-forecasting_data/malaria/hybrid_deliverable/lsae_1285",
        "20260720",
    ) in currents
    labels = {lab for r in rows for lab in r["labels"]}
    assert {
        "first_submission",
        "selected_2026_07_31",
        "goalkeepers_2026",
        "goalkeepers_2026_original",
        "goalkeepers_2025",
    } <= labels
    assert sum(1 for r in rows if "first_submission" in r["labels"]) == 15  # 9 pre_restructure nodes + dah, hierarchy, cfr, 2 x lsae_1209 forecasts, upload 2025_08_28
    assert not any(
        "__anchor_first_submission" in r["node"] for r in rows
    )  # the DROP arm is not registered
    assert len([r for r in rows if r["node"].startswith("05-products/malaria")]) == 9


def test_legacy_models_have_one_current(mod):
    assert sum(e.current for e in mod.LEGACY_MODELS) == 1
    assert {e.directory for e in mod.LEGACY_MODELS} == {
        "2026_07_31_full_model_selection_results",
        "2026_07_20_hybrid_fghjul",
        "2025_07_08",
    }


def test_every_entry_has_a_description(mod):
    assert all(r["description"].strip() for r in mod.plan_rows())
