"""
Tests for lib/io/yaml.py

Uses a temporary YAML file with a COVARIATE_DICT block — no project data.
"""

import pytest
import yaml
from pathlib import Path

from idd_forecast_mbp.lib.io.yaml import load_yaml_dictionary, parse_yaml_dictionary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def yaml_file(tmp_path):
    """Write a minimal COVARIATE_DICT YAML to tmp_path and return its path."""
    data = {
        'COVARIATE_DICT': {
            'malaria_pfpr': {
                'covariate_name': 'malaria_pfpr_mean',
                'covariate_resolution_numerator': 1,
                'covariate_resolution_denominator': 12,
                'year_start': 2000,
                'year_end': 2022,
                'synoptic': False,
                'cc_sensitive': False,
                'summary_statistic': 'mean',
                'path': '/some/path',
            },
        }
    }
    p = tmp_path / 'COVARIATE_DICT.yaml'
    p.write_text(yaml.dump(data))
    return p


# ---------------------------------------------------------------------------
# load_yaml_dictionary
# ---------------------------------------------------------------------------

def test_load_yaml_dictionary_returns_dict(yaml_file):
    result = load_yaml_dictionary(str(yaml_file))
    assert isinstance(result, dict)


def test_load_yaml_dictionary_contains_covariate(yaml_file):
    result = load_yaml_dictionary(str(yaml_file))
    assert 'malaria_pfpr' in result


def test_load_yaml_dictionary_wrong_key_raises(tmp_path):
    """File missing COVARIATE_DICT key should raise KeyError."""
    p = tmp_path / 'bad.yaml'
    p.write_text(yaml.dump({'OTHER_KEY': {}}))
    with pytest.raises(KeyError):
        load_yaml_dictionary(str(p))


# ---------------------------------------------------------------------------
# parse_yaml_dictionary
# ---------------------------------------------------------------------------

def test_parse_yaml_dictionary_returns_expected_keys(yaml_file, monkeypatch):
    """parse_yaml_dictionary should return a dict with the required keys."""
    import idd_forecast_mbp.lib.io.yaml as yaml_mod
    # Patch the YAML_PATH construction to use our temp file
    monkeypatch.setattr(
        'idd_forecast_mbp.lib.io.yaml.load_yaml_dictionary',
        lambda _: load_yaml_dictionary(str(yaml_file)),
    )
    result = parse_yaml_dictionary('malaria_pfpr')
    expected_keys = {'covariate_name', 'covariate_resolution', 'years', 'synoptic',
                     'cc_sensitive', 'summary_statistic', 'path'}
    assert expected_keys == set(result.keys())


def test_parse_yaml_dictionary_years_list(yaml_file, monkeypatch):
    monkeypatch.setattr(
        'idd_forecast_mbp.lib.io.yaml.load_yaml_dictionary',
        lambda _: load_yaml_dictionary(str(yaml_file)),
    )
    result = parse_yaml_dictionary('malaria_pfpr')
    assert result['years'] == list(range(2000, 2023))


def test_parse_yaml_dictionary_resolution(yaml_file, monkeypatch):
    monkeypatch.setattr(
        'idd_forecast_mbp.lib.io.yaml.load_yaml_dictionary',
        lambda _: load_yaml_dictionary(str(yaml_file)),
    )
    result = parse_yaml_dictionary('malaria_pfpr')
    assert result['covariate_resolution'] == pytest.approx(1 / 12)


def test_parse_yaml_dictionary_missing_covariate_raises(yaml_file, monkeypatch):
    monkeypatch.setattr(
        'idd_forecast_mbp.lib.io.yaml.load_yaml_dictionary',
        lambda _: load_yaml_dictionary(str(yaml_file)),
    )
    with pytest.raises(ValueError, match="not found in the dictionary"):
        parse_yaml_dictionary('nonexistent_covariate')
