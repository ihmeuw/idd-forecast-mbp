"""The fitted-models node and the working-slot write paths in constants."""

from __future__ import annotations

import pytest

from idd_forecast_mbp import constants as mbpc


def _node(tmp_path):
    node = tmp_path / "models"
    snap = node / "20260916"
    snap.mkdir(parents=True)
    (snap / mbpc.MAL_MODELS_RDATA).write_bytes(b"rdata")
    (node / "current").symlink_to("20260916")
    (node / "full_model_selection_results").symlink_to("20260916")
    return node, snap


def test_malaria_model_dir_resolves_current_name_and_label(tmp_path):
    node, snap = _node(tmp_path)
    assert mbpc.malaria_model_dir(node=node) == snap.resolve()
    assert mbpc.malaria_model_dir("20260916", node=node) == snap.resolve()
    assert (
        mbpc.malaria_model_dir("full_model_selection_results", node=node)
        == snap.resolve()
    )


def test_malaria_model_dir_refuses_missing(tmp_path):
    node, snap = _node(tmp_path)
    with pytest.raises(FileNotFoundError, match="no malaria model 'nope'"):
        mbpc.malaria_model_dir("nope", node=node)
    (snap / mbpc.MAL_MODELS_RDATA).unlink()
    with pytest.raises(FileNotFoundError, match=r"has no malaria_models\.RData"):
        mbpc.malaria_model_dir(node=node)
    with pytest.raises(FileNotFoundError):
        mbpc.malaria_model_dir(node=tmp_path / "absent")


def test_every_write_slot_is_working():
    names = [n for n in dir(mbpc) if n.endswith("_WRITE_PATH")]
    assert names, "no *_WRITE_PATH constants found"
    for name in names:
        assert getattr(mbpc, name).name == "working", name
    assert mbpc.pixel_write_path("lsae_1285").name == "working"
    assert mbpc.mal_products_write_path("some_run").name == "working"
    assert mbpc.FORECASTING_DATA_PATH.name == "working"
    assert mbpc.MAL_MODELS_READ_PATH.name == "current"
