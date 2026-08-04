"""Tests for :mod:`idd_forecast_mbp.lib.data.dengue_inputs`.

Only the pure transforms are covered here — the reads pull 728 MB / 1.6 GB
cluster artifacts and belong in a QC check, not a unit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.data.dengue_inputs import (
    URBAN_COLUMN,
    attach_age_sex_rr,
    build_age_sex_rr,
    derive_model_columns,
    load_dengue_inputs,
)

ANCHOR = 2023


@pytest.fixture
def past_rows():
    """Three rows exercising the interesting response edges."""
    return pd.DataFrame({
        URBAN_COLUMN: [0.0, 0.5, 1.0],
        "gdppc_mean": [1000.0, 2000.0, 4000.0],
        "dengue_inc_rate": [0.02, 0.0, 0.01],      # middle row: log -> -inf
        "dengue_mort_rate": [0.0002, 0.0, 0.02],   # last row: cfr == 2 -> outside (0,1)
    })


class TestDeriveModelColumns:
    def test_urban_is_clipped_off_the_boundaries(self, past_rows):
        out = derive_model_columns(past_rows)
        assert out["urban_fraction"].min() > 0.0
        assert out["urban_fraction"].max() < 1.0

    def test_log_gdppc(self, past_rows):
        out = derive_model_columns(past_rows)
        np.testing.assert_allclose(out["log_gdppc_mean"], np.log([1000.0, 2000.0, 4000.0]))

    def test_zero_incidence_becomes_negative_infinity(self, past_rows):
        """Not repaired — an infinite response drops the row at fit time."""
        out = derive_model_columns(past_rows)
        assert np.isneginf(out.loc[1, "log_dengue_inc_rate"])
        assert np.isfinite(out.loc[0, "log_dengue_inc_rate"])

    def test_cfr_only_where_incidence_is_positive(self, past_rows):
        out = derive_model_columns(past_rows)
        assert out.loc[0, "dengue_cfr"] == pytest.approx(0.01)
        assert np.isnan(out.loc[1, "dengue_cfr"])

    def test_logit_cfr_only_on_the_open_interval(self, past_rows):
        out = derive_model_columns(past_rows)
        assert np.isfinite(out.loc[0, "logit_dengue_cfr"])
        assert np.isnan(out.loc[1, "logit_dengue_cfr"])   # cfr is NaN
        assert np.isnan(out.loc[2, "logit_dengue_cfr"])   # cfr == 2, outside (0,1)

    def test_log_mortality_rate_for_the_direct_outcome_structures(self, past_rows):
        """Formulations that regress mortality directly need this response."""
        out = derive_model_columns(past_rows)
        assert out.loc[0, "log_dengue_mort_rate"] == pytest.approx(np.log(0.0002))
        assert np.isnan(out.loc[1, "log_dengue_mort_rate"])

    def test_logit_urban_is_provided_for_the_2025_spec(self, past_rows):
        """final_models_dengue.r enters urban as a logit, not a fraction."""
        out = derive_model_columns(past_rows)
        assert "logit_urban_fraction" in out.columns
        assert np.isfinite(out["logit_urban_fraction"]).all()
        # monotone in the fraction, and finite at both clipped boundaries
        assert out["logit_urban_fraction"].is_monotonic_increasing

    def test_input_is_not_mutated(self, past_rows):
        before = past_rows.copy()
        derive_model_columns(past_rows)
        pd.testing.assert_frame_equal(past_rows, before)


@pytest.fixture
def observed_age_sex():
    """Two locations x two age/sex cells at the anchor year, plus a decoy year.

    Location 10 has a positive reference count; location 20 has zero, so it must
    be excluded from the base set.
    """
    rows = []
    for loc, base_rate, base_count in ((10, 0.01, 5.0), (20, 0.02, 0.0)):
        rows += [
            {"location_id": loc, "year_id": ANCHOR, "age_group_id": 3, "sex_id": 1,
             "dengue_inc_rate": base_rate, "dengue_inc_count": base_count},
            {"location_id": loc, "year_id": ANCHOR, "age_group_id": 4, "sex_id": 2,
             "dengue_inc_rate": base_rate * 3, "dengue_inc_count": 9.0},
            {"location_id": loc, "year_id": 1999, "age_group_id": 3, "sex_id": 1,
             "dengue_inc_rate": 99.0, "dengue_inc_count": 99.0},
        ]
    return pd.DataFrame(rows)


class TestBuildAgeSexRr:
    def test_reference_cell_has_unit_relative_risk(self, observed_age_sex):
        rr, _ = build_age_sex_rr(observed_age_sex, anchor_year=ANCHOR,
                                 reference_age_group_id=3, reference_sex_id=1)
        ref = rr[(rr.age_group_id == 3) & (rr.sex_id == 1)]
        assert ref["rr_inc_as"].tolist() == [1.0]

    def test_relative_risk_is_against_the_reference_cell(self, observed_age_sex):
        rr, _ = build_age_sex_rr(observed_age_sex, anchor_year=ANCHOR,
                                 reference_age_group_id=3, reference_sex_id=1)
        other = rr[(rr.age_group_id == 4) & (rr.sex_id == 2)]
        assert other["rr_inc_as"].tolist() == [pytest.approx(3.0)]

    def test_location_without_a_reference_count_is_excluded(self, observed_age_sex):
        _, base_ids = build_age_sex_rr(observed_age_sex, anchor_year=ANCHOR,
                                       reference_age_group_id=3, reference_sex_id=1)
        assert base_ids.tolist() == [10]

    def test_only_the_anchor_year_is_used(self, observed_age_sex):
        rr, _ = build_age_sex_rr(observed_age_sex, anchor_year=ANCHOR,
                                 reference_age_group_id=3, reference_sex_id=1)
        assert 99.0 not in rr["rr_inc_as"].to_numpy()

    def test_a_different_reference_cell_rebases_the_relative_risks(self, observed_age_sex):
        """The reference group is a parameter, so choosing another cell rebases rr."""
        rr, base_ids = build_age_sex_rr(observed_age_sex, anchor_year=ANCHOR,
                                        reference_age_group_id=4, reference_sex_id=2)
        assert base_ids.tolist() == [10, 20]  # both have a positive count in that cell
        ref = rr[(rr.age_group_id == 4) & (rr.sex_id == 2) & (rr.fhs_location_id == 10)]
        other = rr[(rr.age_group_id == 3) & (rr.sex_id == 1) & (rr.fhs_location_id == 10)]
        assert ref["rr_inc_as"].tolist() == [1.0]
        assert other["rr_inc_as"].tolist() == [pytest.approx(1 / 3)]

    def test_output_columns(self, observed_age_sex):
        rr, _ = build_age_sex_rr(observed_age_sex, anchor_year=ANCHOR,
                                 reference_age_group_id=3, reference_sex_id=1)
        assert list(rr.columns) == ["fhs_location_id", "age_group_id", "sex_id", "rr_inc_as"]


class TestAttachAgeSexRr:
    @pytest.fixture
    def past(self):
        """Two admin-2 units under FHS parent 10, one under excluded parent 20."""
        rows = []
        for loc, fhs, a0 in ((101, 10, 163), (102, 10, 163), (201, 20, 11)):
            for age, sex in ((3, 1), (4, 2)):
                rows.append({"location_id": loc, "fhs_location_id": fhs,
                             "A0_location_id": a0, "age_group_id": age, "sex_id": sex,
                             "year_id": ANCHOR})
        return pd.DataFrame(rows)

    @pytest.fixture
    def rr(self):
        return pd.DataFrame([
            {"fhs_location_id": 10, "age_group_id": 3, "sex_id": 1, "rr_inc_as": 1.0},
            {"fhs_location_id": 10, "age_group_id": 4, "sex_id": 2, "rr_inc_as": 3.0},
        ])

    def test_locations_without_a_base_parent_are_dropped(self, past, rr):
        out = attach_age_sex_rr(past, rr, [10])
        assert set(out["location_id"]) == {101, 102}

    def test_children_inherit_their_fhs_parents_pattern(self, past, rr):
        out = attach_age_sex_rr(past, rr, [10])
        for loc in (101, 102):
            cell = out[(out.location_id == loc) & (out.age_group_id == 4)]
            assert cell["rr_inc_as"].tolist() == [3.0]

    def test_country_codes_are_contiguous_integers(self, past, rr):
        out = attach_age_sex_rr(past, rr, [10, 20])
        assert sorted(out["A0_af"].unique()) == [0, 1]
        assert pd.api.types.is_integer_dtype(out["A0_af"])

    def test_country_code_is_stable_within_a_country(self, past, rr):
        out = attach_age_sex_rr(past, rr, [10, 20])
        assert out.groupby("A0_location_id")["A0_af"].nunique().max() == 1

    def test_age_sex_factor_code_is_assigned(self, past, rr):
        """as_id must come from the loader, not be re-derived per notebook.

        A factor coding that differs between fit and predict silently reassigns
        effects, so there can only be one place that builds it.
        """
        out = attach_age_sex_rr(past, rr, [10, 20])
        assert "as_id" in out.columns
        assert pd.api.types.is_integer_dtype(out["as_id"])
        # one code per distinct (age, sex) cell, stable across locations
        per_cell = out.groupby(["age_group_id", "sex_id"])["as_id"].nunique()
        assert (per_cell == 1).all()
        assert out["as_id"].nunique() == out.groupby(
            ["age_group_id", "sex_id"]).ngroups

    def test_input_is_not_mutated(self, past, rr):
        before = past.copy()
        attach_age_sex_rr(past, rr, [10])
        pd.testing.assert_frame_equal(past, before)


class TestLoadDengueInputs:
    def test_rejects_unknown_grain(self):
        with pytest.raises(ValueError, match="grain must be one of"):
            load_dengue_inputs(grain="admin_7")

    @pytest.fixture
    def artifacts(self, tmp_path):
        """Miniature stand-ins for the four stage-02 parquets.

        Locations 10 and 20 are FHS-most-detailed; 101 is an admin-2 child of 10
        and must be filtered out by a ``grain='fhs'`` read.
        """
        hierarchy = pd.DataFrame([
            {"location_id": 1, "fhs_location_id": 1, "level": 0,
             "most_detailed_fhs": 0, "most_detailed_lsae": 0},
            {"location_id": 10, "fhs_location_id": 10, "level": 3,
             "most_detailed_fhs": 1, "most_detailed_lsae": 0},
            {"location_id": 20, "fhs_location_id": 20, "level": 3,
             "most_detailed_fhs": 1, "most_detailed_lsae": 0},
            {"location_id": 101, "fhs_location_id": 10, "level": 5,
             "most_detailed_fhs": 0, "most_detailed_lsae": 1},
        ])

        past_rows = []
        for loc in (10, 20, 101):
            for age, sex in ((3, 1), (4, 2)):
                past_rows.append({
                    "location_id": loc, "year_id": ANCHOR, "A0_location_id": 163,
                    "age_group_id": age, "sex_id": sex,
                    URBAN_COLUMN: 0.4, "gdppc_mean": 1500.0,
                    "dengue_inc_rate": 0.01 * (3 if age == 4 else 1),
                    "dengue_mort_rate": 0.0001,
                })

        observed_rows = []
        for loc in (10, 20):
            observed_rows += [
                {"location_id": loc, "year_id": ANCHOR, "age_group_id": 3, "sex_id": 1,
                 "dengue_inc_rate": 0.01, "dengue_inc_count": 5.0},
                {"location_id": loc, "year_id": ANCHOR, "age_group_id": 4, "sex_id": 2,
                 "dengue_inc_rate": 0.03, "dengue_inc_count": 9.0},
            ]

        paths = {}
        for name, frame in (
            ("past", pd.DataFrame(past_rows)),
            ("observed_as", pd.DataFrame(observed_rows)),
            ("observed_aa", pd.DataFrame([{"location_id": 1, "year_id": ANCHOR,
                                           "dengue_inc_rate": 0.01}])),
            ("population", pd.DataFrame([{"location_id": 1, "year_id": ANCHOR,
                                          "population": 100.0}])),
        ):
            p = tmp_path / f"{name}.parquet"
            frame.to_parquet(p)
            paths[name] = p
        return hierarchy, paths

    def _load(self, monkeypatch, hierarchy, paths, **kwargs):
        monkeypatch.setattr(
            "idd_forecast_mbp.lib.data.dengue_inputs.load_hierarchy",
            lambda *a, **k: hierarchy,
        )
        return load_dengue_inputs(
            past_inputs_path=paths["past"],
            observed_age_sex_path=paths["observed_as"],
            observed_all_age_path=paths["observed_aa"],
            population_path=paths["population"],
            anchor_year=ANCHOR,
            **kwargs,
        )

    def test_fhs_grain_excludes_admin_2_rows(self, monkeypatch, artifacts):
        hierarchy, paths = artifacts
        got = self._load(monkeypatch, hierarchy, paths, grain="fhs")
        assert set(got.fit_frame["location_id"]) == {10, 20}

    def test_lsae_grain_selects_the_admin_2_rows(self, monkeypatch, artifacts):
        hierarchy, paths = artifacts
        got = self._load(monkeypatch, hierarchy, paths, grain="lsae")
        assert set(got.fit_frame["location_id"]) == {101}

    def test_derived_columns_and_relative_risk_are_attached(self, monkeypatch, artifacts):
        hierarchy, paths = artifacts
        got = self._load(monkeypatch, hierarchy, paths)
        for col in ("log_dengue_inc_rate", "logit_dengue_cfr", "urban_fraction",
                    "rr_inc_as", "A0_af"):
            assert col in got.fit_frame.columns
        older = got.fit_frame[got.fit_frame.age_group_id == 4]
        assert older["rr_inc_as"].unique().tolist() == [pytest.approx(3.0)]

    def test_metadata_is_carried_through(self, monkeypatch, artifacts):
        hierarchy, paths = artifacts
        got = self._load(monkeypatch, hierarchy, paths)
        assert got.grain == "fhs"
        assert got.anchor_year == ANCHOR
        assert got.reference_age_group_id == 3
        assert got.reference_sex_id == 1
        assert sorted(got.base_location_ids.tolist()) == [10, 20]

    def test_reference_group_is_overridable(self, monkeypatch, artifacts):
        hierarchy, paths = artifacts
        got = self._load(monkeypatch, hierarchy, paths,
                         reference_age_group_id=4, reference_sex_id=2)
        assert got.reference_age_group_id == 4
        base_cells = got.fit_frame[(got.fit_frame.age_group_id == 4)
                                   & (got.fit_frame.sex_id == 2)]
        assert base_cells["rr_inc_as"].unique().tolist() == [1.0]
