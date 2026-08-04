"""Tests for the malaria model registry reader in idd_forecast_mbp.constants.

The registry is written by 03_modeling/02_fit_final_malaria_models.r (via
lib/model_registry.R); these tests exercise the Python read side, which is what
the forecasting/Python pipeline uses to resolve a model run_date.
"""
import json

import pytest

from idd_forecast_mbp import constants as rfc


def _write_registry(tmp_path, records):
    path = tmp_path / "malaria_model_registry.json"
    path.write_text(json.dumps(records, indent=2))
    return path


def test_get_best_run_date(tmp_path):
    path = _write_registry(tmp_path, [
        {"run_date": "2026_06_01", "best": True, "description": "x"},
        {"run_date": "2026_05_01", "best": False, "description": "y"},
    ])
    assert rfc.get_malaria_model_run_date(best=True, path=path) == "2026_06_01"


def test_get_by_explicit_run_date(tmp_path):
    path = _write_registry(tmp_path, [
        {"run_date": "2026_06_01", "best": True, "description": "x"},
        {"run_date": "2026_05_01", "best": False, "description": "y"},
    ])
    # An explicit run_date is honored even if it is not the best.
    assert rfc.get_malaria_model_run_date(run_date="2026_05_01", path=path) == "2026_05_01"


def test_read_registry_returns_records(tmp_path):
    records = [{"run_date": "2026_06_01", "best": True, "description": "x"}]
    path = _write_registry(tmp_path, records)
    assert rfc.read_malaria_model_registry(path) == records


def test_read_missing_registry_returns_empty(tmp_path):
    assert rfc.read_malaria_model_registry(tmp_path / "nope.json") == []


def test_missing_run_date_raises(tmp_path):
    path = _write_registry(tmp_path, [
        {"run_date": "2026_06_01", "best": True, "description": "x"},
    ])
    with pytest.raises(KeyError):
        rfc.get_malaria_model_run_date(run_date="1999_01_01", path=path)


def test_no_best_raises(tmp_path):
    path = _write_registry(tmp_path, [
        {"run_date": "2026_05_01", "best": False, "description": "y"},
    ])
    with pytest.raises(ValueError):
        rfc.get_malaria_model_run_date(best=True, path=path)


def test_multiple_best_raises(tmp_path):
    path = _write_registry(tmp_path, [
        {"run_date": "2026_06_01", "best": True, "description": "x"},
        {"run_date": "2026_05_01", "best": True, "description": "y"},
    ])
    with pytest.raises(ValueError):
        rfc.get_malaria_model_run_date(best=True, path=path)


def test_empty_or_missing_registry_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rfc.get_malaria_model_run_date(best=True, path=tmp_path / "nope.json")
