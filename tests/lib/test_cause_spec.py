"""Tests for the cause specification.

Emphasis is on the properties that were argued over for three rounds and settled deliberately
(see ``.claude/DENGUE_CONSULT_ROUND3.md``), because those are the ones a future refactor is most
likely to undo:

* alternative covariate futures are NOT representable here — they belong to a run
* ``as_draw_persist_grain`` is a grain and is independent of ``fit_grain``
* a point anchor and a window-mean anchor imply *different* correctness checks
* the burden filter is one rule with one parameter, not a taxonomy
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from idd_forecast_mbp.lib.cause_spec import (
    AnchorKind,
    AnchorSpec,
    BurdenFilter,
    CauseSpec,
    Grain,
    MeasureSpec,
    MeasureStructure,
    get_cause_spec,
    known_causes,
)

# --------------------------------------------------------------------------- Grain


def test_admin2_is_uniform_level_5_and_fhs_is_mixed():
    assert Grain.ADMIN2.uniform_level == 5
    assert not Grain.ADMIN2.is_mixed_level
    assert Grain.FHS_MOST_DETAILED.uniform_level is None
    assert Grain.FHS_MOST_DETAILED.is_mixed_level


# --------------------------------------------------------------------------- AnchorSpec


def test_point_anchor_reproduces_observed_and_window_mean_does_not():
    """The distinction that makes the anchor check correct rather than misleading."""
    point = AnchorSpec(AnchorKind.POINT, (2023,))
    window = AnchorSpec(AnchorKind.WINDOW_MEAN, tuple(range(2014, 2024)))
    assert point.reproduces_observed is True
    assert window.reproduces_observed is False


def test_point_anchor_rejects_multiple_years():
    with pytest.raises(ValueError, match="exactly one year"):
        AnchorSpec(AnchorKind.POINT, (2022, 2023))


def test_window_mean_rejects_a_single_year():
    with pytest.raises(ValueError, match="at least two years"):
        AnchorSpec(AnchorKind.WINDOW_MEAN, (2023,))


def test_anchor_rejects_empty_and_duplicate_years():
    with pytest.raises(ValueError, match="must not be empty"):
        AnchorSpec(AnchorKind.POINT, ())
    with pytest.raises(ValueError, match="duplicates"):
        AnchorSpec(AnchorKind.WINDOW_MEAN, (2020, 2020, 2021))


def test_anchor_rejects_nonpositive_outlier_sd():
    with pytest.raises(ValueError, match="outlier_sd must be positive"):
        AnchorSpec(AnchorKind.WINDOW_MEAN, (2020, 2021), outlier_sd=0.0)


def _observed() -> pd.DataFrame:
    # Location 1 spikes in 2023; location 2 is flat. The spike is the dengue-2023 situation.
    return pd.DataFrame(
        {
            "location_id": [1, 1, 1, 2, 2, 2],
            "year_id": [2021, 2022, 2023, 2021, 2022, 2023],
            "value": [10.0, 20.0, 90.0, 5.0, 5.0, 5.0],
        }
    )


def test_point_anchor_target_is_that_years_observed_value():
    spec = AnchorSpec(AnchorKind.POINT, (2023,))
    t = spec.target(_observed(), "value").set_index("location_id")
    assert t.loc[1, "anchor_target"] == pytest.approx(90.0)
    assert t.loc[2, "anchor_target"] == pytest.approx(5.0)
    assert set(t.n_years) == {1}


def test_window_mean_target_is_the_window_mean_not_the_last_year():
    """The bug this prevents: comparing a window-mean run against observed-at-anchor-year."""
    spec = AnchorSpec(AnchorKind.WINDOW_MEAN, (2021, 2022, 2023))
    t = spec.target(_observed(), "value").set_index("location_id")
    assert t.loc[1, "anchor_target"] == pytest.approx(40.0)   # (10+20+90)/3, NOT 90
    assert t.loc[2, "anchor_target"] == pytest.approx(5.0)
    assert set(t.n_years) == {3}


def test_window_mean_target_degrades_on_a_partial_window():
    """A thin target must be visible via n_years rather than silently NaN."""
    spec = AnchorSpec(AnchorKind.WINDOW_MEAN, (2019, 2020, 2021, 2022, 2023))
    t = spec.target(_observed(), "value").set_index("location_id")
    assert set(t.n_years) == {3}          # only 2021-2023 present
    assert t.loc[1, "anchor_target"] == pytest.approx(40.0)


def test_anchor_target_raises_when_no_year_is_present():
    spec = AnchorSpec(AnchorKind.POINT, (1990,))
    with pytest.raises(ValueError, match="no observed rows in anchor years"):
        spec.target(_observed(), "value")


def test_anchor_labels_describe_the_semantics():
    assert AnchorSpec(AnchorKind.POINT, (2023,)).label == "anchored to observed 2023"
    lbl = AnchorSpec(AnchorKind.WINDOW_MEAN, tuple(range(2014, 2024))).label
    assert lbl == "anchored to the 2014–2023 observed mean"


# --------------------------------------------------------------------------- BurdenFilter


@pytest.fixture
def counts() -> pd.DataFrame:
    return pd.DataFrame({"location_id": [1, 2, 3], "inc_count": [0.0, 5.0, 100.0]})


def test_nonstrict_zero_threshold_is_inert(counts):
    """Malaria's filter: >= 0 on a non-negative count keeps everything."""
    f = BurdenFilter("inc_count", threshold=0.0, strict=False)
    assert f.is_inert
    assert len(f.apply(counts)) == 3


def test_strict_zero_threshold_drops_zero_burden(counts):
    """Dengue's filter: > 0 drops the no-burden locations."""
    f = BurdenFilter("inc_count", threshold=0.0, strict=True)
    assert not f.is_inert
    assert sorted(f.apply(counts).location_id) == [2, 3]


def test_strictness_is_the_only_difference_between_the_two_causes(counts):
    """Both causes' rules differ in exactly one bool -- no taxonomy needed."""
    lenient = BurdenFilter("inc_count", 0.0, strict=False).apply(counts)
    strict = BurdenFilter("inc_count", 0.0, strict=True).apply(counts)
    assert set(lenient.location_id) - set(strict.location_id) == {1}


def test_positive_threshold_is_not_inert(counts):
    f = BurdenFilter("inc_count", threshold=10.0, strict=False)
    assert not f.is_inert
    assert sorted(f.apply(counts).location_id) == [3]


def test_missing_column_raises_rather_than_silently_keeping_everything(counts):
    with pytest.raises(KeyError, match="not in frame"):
        BurdenFilter("nope").mask(counts)


def test_filter_description_round_trips_the_comparison():
    assert BurdenFilter("c", 0.0, strict=True).description == "c > 0"
    assert BurdenFilter("c", 2.5, strict=False).description == "c >= 2.5"


# --------------------------------------------------------------------------- Measures


def test_independent_measures_order_deterministically():
    ms = MeasureStructure(
        (
            MeasureSpec("incidence", "inc_count", "inc_rate"),
            MeasureSpec("mortality", "mort_count", "mort_rate"),
        )
    )
    assert ms.order() == ("incidence", "mortality")
    assert ms.names == ("incidence", "mortality")
    assert len(ms) == 2


def test_feed_forward_dependency_orders_producer_before_consumer():
    """Dengue predicts mortality first, then uses it as a covariate for incidence."""
    ms = MeasureStructure(
        (
            MeasureSpec("incidence", "inc_count", "inc_rate", depends_on=("mortality",)),
            MeasureSpec("mortality", "mort_count", "mort_rate"),
        )
    )
    order = ms.order()
    assert order.index("mortality") < order.index("incidence")


def test_intermediate_measures_are_excluded_from_deliverables():
    ms = MeasureStructure(
        (
            MeasureSpec("cfr", None, None, intermediate=True),
            MeasureSpec("incidence", "inc_count", "inc_rate"),
            MeasureSpec("mortality", "mort_count", "mort_rate", depends_on=("cfr", "incidence")),
        )
    )
    assert [m.name for m in ms.deliverable] == ["incidence", "mortality"]
    order = ms.order()
    assert order.index("cfr") < order.index("mortality")
    assert order.index("incidence") < order.index("mortality")


def test_cyclic_dependencies_fail_at_construction_not_first_use():
    with pytest.raises(ValueError, match="cyclic"):
        MeasureStructure(
            (
                MeasureSpec("a", "ac", "ar", depends_on=("b",)),
                MeasureSpec("b", "bc", "br", depends_on=("a",)),
            )
        )


def test_dependency_on_unknown_measure_is_rejected():
    with pytest.raises(ValueError, match="unknown measure"):
        MeasureStructure((MeasureSpec("a", "ac", "ar", depends_on=("ghost",)),))


def test_duplicate_measure_names_rejected():
    with pytest.raises(ValueError, match="duplicate measure names"):
        MeasureStructure(
            (
                MeasureSpec("a", "ac", "ar"),
                MeasureSpec("a", "ac2", "ar2"),
            )
        )


def test_deliverable_measure_requires_both_count_and_rate():
    with pytest.raises(ValueError, match="needs both count_col and rate_col"):
        MeasureSpec("incidence", "inc_count", None)


def test_measure_cannot_depend_on_itself():
    with pytest.raises(ValueError, match="depends on itself"):
        MeasureSpec("a", "ac", "ar", depends_on=("a",))


def test_lookup_by_name_and_missing_name():
    ms = MeasureStructure((MeasureSpec("incidence", "inc_count", "inc_rate"),))
    assert ms["incidence"].count_col == "inc_count"
    with pytest.raises(KeyError, match="no measure named"):
        ms["mortality"]


# --------------------------------------------------------------------------- CauseSpec


def test_both_causes_build():
    assert known_causes() == ("dengue", "malaria")
    for name in known_causes():
        spec = get_cause_spec(name)
        assert spec.name == name
        assert spec.burden_column


def test_unknown_cause_raises():
    with pytest.raises(ValueError, match="unknown cause"):
        get_cause_spec("chikungunya")


def test_cause_lookup_is_case_insensitive():
    assert get_cause_spec("MALARIA").name == "malaria"


def test_malaria_and_dengue_differ_only_by_parameters():
    mal, den = get_cause_spec("malaria"), get_cause_spec("dengue")
    assert mal.fit_grain is Grain.ADMIN2
    assert den.fit_grain is Grain.FHS_MOST_DETAILED
    assert mal.burden_column == "malaria_inc_count"
    assert den.burden_column == "dengue_inc_count"
    assert mal.absent_means_zero is False
    assert den.absent_means_zero is True


def test_no_field_holds_a_refit_axis():
    """Anything that forces a refit varies per run, so it cannot be cause-level.

    Guards the extensibility property: adding a covariate, a suitability variant or a candidate
    formulation must never require editing CauseSpec.
    """
    fields = set(CauseSpec.__dataclass_fields__)
    refit_axes = {
        "covariates",
        "anchor",
        "measures",
        "formulation",
        "engine",
        "model_id",
        "suitability_variant",
        "reference_age_group_id",
        "reference_sex_id",
        "burden_filter",       # the threshold is a run choice; only the column is cause-level
        "burden_threshold",
        "year_center",
    }
    assert not (fields & refit_axes), f"refit axes must live in the run spec: {fields & refit_axes}"


def test_burden_column_is_cause_level_but_the_threshold_is_not():
    """Both causes filter on their own count column; the comparison is supplied per run."""
    mal = get_cause_spec("malaria")
    inert = mal.default_burden_filter(0.0, strict=False)      # malaria's actual run setting
    strict = mal.default_burden_filter(0.0, strict=True)      # a different run could ask for this
    assert inert.column == strict.column == "malaria_inc_count"
    assert inert.is_inert
    assert not strict.is_inert


def test_as_draw_persist_grain_is_a_grain_and_dengue_is_not_none():
    """The has_dah mistake at one level down: a boolean made dengue's case unnameable."""
    den = get_cause_spec("dengue")
    assert den.as_draw_persist_grain is Grain.FHS_MOST_DETAILED
    assert den.as_draw_persist_grain is not None
    assert get_cause_spec("malaria").as_draw_persist_grain is Grain.ADMIN2


def test_persist_grain_is_independent_of_fit_grain():
    """A cause may fit at one grain and persist at another; nothing may couple them."""
    base = get_cause_spec("dengue")
    crossed = CauseSpec(
        name="crossed",
        cause_id=999,
        fit_grain=Grain.FHS_MOST_DETAILED,
        burden_column="x_count",
        absent_means_zero=True,
        raked_aa_read_path=base.raked_aa_read_path,
        past_inputs_path=base.past_inputs_path,
        forecast_inputs_path=base.forecast_inputs_path,
        products_read_path=base.products_read_path,
        as_draw_persist_grain=Grain.ADMIN2,        # deliberately != fit_grain
    )
    assert crossed.fit_grain is Grain.FHS_MOST_DETAILED
    assert crossed.as_draw_persist_grain is Grain.ADMIN2


def test_no_field_can_express_a_covariate_future():
    """Alternative trajectories belong to a run, so no CauseSpec field may name one.

    Guards against re-adding has_dah / secondary_axis / AxisSpec under any name.
    """
    fields = set(CauseSpec.__dataclass_fields__)
    banned = {
        "has_dah",
        "secondary_axis",
        "axis",
        "axis_spec",
        "dah_scenarios",
        "trajectories",
        "covariate_trajectories",
        "decay",
        "holds",
        "scenarios",
    }
    assert not (fields & banned), f"CauseSpec must not carry run-level variation: {fields & banned}"


def test_empty_identity_fields_rejected():
    base = get_cause_spec("malaria")
    common = {
        "cause_id": 1,
        "fit_grain": Grain.ADMIN2,
        "absent_means_zero": False,
        "raked_aa_read_path": base.raked_aa_read_path,
        "past_inputs_path": base.past_inputs_path,
        "forecast_inputs_path": base.forecast_inputs_path,
        "products_read_path": base.products_read_path,
    }
    with pytest.raises(ValueError, match="name must not be empty"):
        CauseSpec(name="", burden_column="c", **common)
    with pytest.raises(ValueError, match="burden_column must not be empty"):
        CauseSpec(name="x", burden_column="", **common)


def test_product_filename_uses_the_trajectory_token():
    """Token is {trajectory}; {axis_value} was retracted vocabulary."""
    mal = get_cause_spec("malaria")
    assert mal.product_filename("ssp245", "Baseline") == (
        "all_age_summary_ssp245_Baseline.parquet"
    )
    den = get_cause_spec("dengue")
    assert den.product_filename("ssp126", "logistic_k8") == (
        "all_age_summary_ssp126_logistic_k8.parquet"
    )


def test_product_filename_requires_both_parts():
    mal = get_cause_spec("malaria")
    with pytest.raises(ValueError, match="both ssp_scenario and trajectory"):
        mal.product_filename("ssp245", "")


def test_anchor_filename_takes_the_anchor_rather_than_storing_it():
    """The anchor is per-run, so two runs of one cause may name different anchor files."""
    mal = get_cause_spec("malaria")
    point = AnchorSpec(AnchorKind.POINT, (2023,))
    window = AnchorSpec(AnchorKind.WINDOW_MEAN, tuple(range(2014, 2024)))
    assert mal.anchor_filename("ssp245", "Baseline", point) == (
        "anchor_2023_ssp245_Baseline.parquet"
    )
    assert mal.anchor_filename("ssp245", "Baseline", window) == (
        "anchor_2014_ssp245_Baseline.parquet"
    )


def test_mixed_level_flag_drives_the_rollup_choice():
    """finish_run must use roll_up_to_ancestors, not a fixed start_level, when True."""
    assert get_cause_spec("dengue").uses_mixed_level_leaves is True
    assert get_cause_spec("malaria").uses_mixed_level_leaves is False


def test_specs_are_frozen():
    mal = get_cause_spec("malaria")
    with pytest.raises(FrozenInstanceError):
        mal.name = "nope"  # type: ignore[misc]


def test_adding_a_covariate_requires_no_cause_spec_change():
    """The extensibility property, stated as a test.

    Covariate metadata lives in lib/io/covariate_registry; which covariates a run uses comes
    from its formulation. Neither is reachable from CauseSpec, so growing either cannot break
    or require touching it.
    """
    fields = set(CauseSpec.__dataclass_fields__)
    assert "covariates" not in fields
    mal = get_cause_spec("malaria")
    assert not any("covariate" in f or "suitab" in f for f in fields)
    # The spec still resolves everything a stage script needs to FIND covariate data.
    assert mal.forecast_inputs_path is not None
    assert mal.past_inputs_path is not None
