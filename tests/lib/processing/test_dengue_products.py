"""Tests for :mod:`idd_forecast_mbp.lib.processing.dengue_products`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.dengue_products import (
    age_sex_rates_to_all_age_counts,
    aggregate_counts,
    build_hierarchy_products,
    counts_to_rates,
    rates_to_counts,
    summarize_products,
)

YEAR = 2023


@pytest.fixture
def hierarchy():
    """global 1 -> country 10 (leaf) and country 20 (leaf)."""
    return pd.DataFrame([
        {"location_id": 1, "parent_id": 0, "level": 0, "path_to_top_parent": "1"},
        {"location_id": 10, "parent_id": 1, "level": 3, "path_to_top_parent": "1,10"},
        {"location_id": 20, "parent_id": 1, "level": 3, "path_to_top_parent": "1,20"},
    ])


@pytest.fixture
def population():
    """Global population is 300 — deliberately more than the two leaves' 100 + 100.

    The extra 100 stands for locations the forecast never modelled. A correct
    aggregate rate divides by 300; summing the modelled children gives 200 and
    inflates the rate by 1.5x.
    """
    return pd.DataFrame([
        {"location_id": 1, "year_id": YEAR, "population": 300.0},
        {"location_id": 10, "year_id": YEAR, "population": 100.0},
        {"location_id": 20, "year_id": YEAR, "population": 100.0},
    ])


@pytest.fixture
def leaf_rates():
    return pd.DataFrame([
        {"location_id": 10, "year_id": YEAR, "dengue_inc_rate": 0.10,
         "dengue_mort_rate": 0.01},
        {"location_id": 20, "year_id": YEAR, "dengue_inc_rate": 0.20,
         "dengue_mort_rate": 0.02},
    ])


class TestRatesToCounts:
    def test_count_is_rate_times_own_population(self, leaf_rates, population):
        got = rates_to_counts(leaf_rates, population).set_index("location_id")
        assert got.loc[10, "dengue_inc_count"] == pytest.approx(10.0)
        assert got.loc[20, "dengue_inc_count"] == pytest.approx(20.0)

    def test_both_measures_are_converted(self, leaf_rates, population):
        got = rates_to_counts(leaf_rates, population).set_index("location_id")
        assert got.loc[10, "dengue_mort_count"] == pytest.approx(1.0)

    def test_location_without_population_is_dropped(self, leaf_rates, population):
        """A rate with no denominator cannot become a count."""
        orphan = pd.concat([leaf_rates, pd.DataFrame([
            {"location_id": 99, "year_id": YEAR, "dengue_inc_rate": 0.5,
             "dengue_mort_rate": 0.05},
        ])], ignore_index=True)
        got = rates_to_counts(orphan, population)
        assert 99 not in set(got["location_id"])

    def test_single_measure_subset(self, leaf_rates, population):
        got = rates_to_counts(leaf_rates, population, measures=["inc"])
        assert "dengue_inc_count" in got.columns
        assert "dengue_mort_count" not in got.columns


class TestAggregateCounts:
    def test_parent_is_the_sum_of_children(self, leaf_rates, population, hierarchy):
        counts = rates_to_counts(leaf_rates, population)
        got = aggregate_counts(counts, hierarchy).set_index("location_id")
        assert got.loc[1, "dengue_inc_count"] == pytest.approx(30.0)

    def test_leaves_are_preserved(self, leaf_rates, population, hierarchy):
        counts = rates_to_counts(leaf_rates, population)
        got = aggregate_counts(counts, hierarchy).set_index("location_id")
        assert got.loc[10, "dengue_inc_count"] == pytest.approx(10.0)

    def test_aggregation_happens_within_each_draw(self, population, hierarchy):
        rows = []
        for draw, scale in ((0, 1.0), (1, 2.0)):
            for loc, rate in ((10, 0.10), (20, 0.20)):
                rows.append({"location_id": loc, "year_id": YEAR, "draw": draw,
                             "dengue_inc_rate": rate * scale,
                             "dengue_mort_rate": 0.0})
        counts = rates_to_counts(pd.DataFrame(rows), population)
        got = aggregate_counts(counts, hierarchy, draw_column="draw")
        root = got[got["location_id"] == 1].set_index("draw")["dengue_inc_count"]
        assert root.loc[0] == pytest.approx(30.0)
        assert root.loc[1] == pytest.approx(60.0)


class TestCountsToRates:
    def test_aggregate_rate_uses_the_levels_own_population(
        self, leaf_rates, population, hierarchy,
    ):
        """The rule that matters: divide by 300, not by the children's 200."""
        counts = rates_to_counts(leaf_rates, population)
        aggregated = aggregate_counts(counts, hierarchy)
        got = counts_to_rates(aggregated, population).set_index("location_id")
        assert got.loc[1, "dengue_inc_rate"] == pytest.approx(30.0 / 300.0)

    def test_summed_child_denominator_would_be_wrong(
        self, leaf_rates, population, hierarchy,
    ):
        """Pin the magnitude of the bug this rule prevents: a 1.5x inflation."""
        counts = rates_to_counts(leaf_rates, population)
        aggregated = aggregate_counts(counts, hierarchy)
        correct = counts_to_rates(aggregated, population)
        correct_rate = correct.set_index("location_id").loc[1, "dengue_inc_rate"]
        summed_children = 30.0 / 200.0
        assert summed_children == pytest.approx(1.5 * correct_rate)

    def test_leaf_rate_round_trips(self, leaf_rates, population, hierarchy):
        counts = rates_to_counts(leaf_rates, population)
        aggregated = aggregate_counts(counts, hierarchy)
        got = counts_to_rates(aggregated, population).set_index("location_id")
        assert got.loc[10, "dengue_inc_rate"] == pytest.approx(0.10)

    def test_missing_denominator_gives_nan_not_a_wrong_number(self, hierarchy):
        counts = pd.DataFrame([
            {"location_id": 10, "year_id": YEAR, "dengue_inc_count": 10.0,
             "dengue_mort_count": 1.0},
        ])
        empty_pop = pd.DataFrame(
            columns=["location_id", "year_id", "population"]).astype(
            {"location_id": "int64", "year_id": "int64", "population": "float64"})
        got = counts_to_rates(counts, empty_pop)
        assert np.isnan(got["dengue_inc_rate"]).all()


class TestDroppedLocations:
    def test_unmodelled_location_is_a_true_zero_in_the_numerator(
        self, population, hierarchy,
    ):
        """Location 20 is absent from the rates entirely.

        Its 100 population still sits in the global denominator, so the global
        rate falls rather than the location silently vanishing from both sides.
        """
        only_ten = pd.DataFrame([
            {"location_id": 10, "year_id": YEAR, "dengue_inc_rate": 0.10,
             "dengue_mort_rate": 0.01},
        ])
        got = build_hierarchy_products(only_ten, population, hierarchy)
        root = got.set_index("location_id").loc[1]
        assert root["dengue_inc_count"] == pytest.approx(10.0)
        assert root["dengue_inc_rate"] == pytest.approx(10.0 / 300.0)


class TestAgeSexRatesToAllAgeCounts:
    @pytest.fixture
    def age_sex_population(self):
        """Location 10's 100 people split 40/60 across two age/sex cells."""
        return pd.DataFrame([
            {"location_id": 10, "year_id": YEAR, "age_group_id": 3, "sex_id": 1,
             "population": 40.0},
            {"location_id": 10, "year_id": YEAR, "age_group_id": 4, "sex_id": 2,
             "population": 60.0},
        ])

    @pytest.fixture
    def age_sex_rates(self):
        return pd.DataFrame([
            {"location_id": 10, "year_id": YEAR, "age_group_id": 3, "sex_id": 1,
             "dengue_inc_rate": 0.10, "dengue_mort_rate": 0.01},
            {"location_id": 10, "year_id": YEAR, "age_group_id": 4, "sex_id": 2,
             "dengue_inc_rate": 0.20, "dengue_mort_rate": 0.02},
        ])

    def test_each_cell_is_costed_at_its_own_population(
        self, age_sex_rates, age_sex_population,
    ):
        got = age_sex_rates_to_all_age_counts(age_sex_rates, age_sex_population)
        # 0.10*40 + 0.20*60 = 4 + 12 = 16
        assert got["dengue_inc_count"].iloc[0] == pytest.approx(16.0)

    def test_collapses_to_one_row_per_location_year(
        self, age_sex_rates, age_sex_population,
    ):
        got = age_sex_rates_to_all_age_counts(age_sex_rates, age_sex_population)
        assert len(got) == 1
        assert "age_group_id" not in got.columns

    def test_using_all_age_population_instead_inflates_by_the_cell_count(
        self, age_sex_rates, population,
    ):
        """Pin the bug: the wrong denominator multiplies the count, badly."""
        correct = age_sex_rates_to_all_age_counts(
            age_sex_rates,
            pd.DataFrame([
                {"location_id": 10, "year_id": YEAR, "age_group_id": 3, "sex_id": 1,
                 "population": 40.0},
                {"location_id": 10, "year_id": YEAR, "age_group_id": 4, "sex_id": 2,
                 "population": 60.0},
            ]),
        )["dengue_inc_count"].iloc[0]
        wrong = rates_to_counts(age_sex_rates, population)["dengue_inc_count"].sum()
        assert correct == pytest.approx(16.0)
        assert wrong == pytest.approx(30.0)   # 0.10*100 + 0.20*100
        assert wrong > correct

    def test_cell_without_population_is_dropped(self, age_sex_rates):
        partial = pd.DataFrame([
            {"location_id": 10, "year_id": YEAR, "age_group_id": 3, "sex_id": 1,
             "population": 40.0},
        ])
        got = age_sex_rates_to_all_age_counts(age_sex_rates, partial)
        assert got["dengue_inc_count"].iloc[0] == pytest.approx(4.0)

    def test_build_hierarchy_products_uses_the_age_sex_path_when_given_one(
        self, age_sex_rates, age_sex_population, population, hierarchy,
    ):
        got = build_hierarchy_products(
            age_sex_rates, population, hierarchy,
            age_sex_population=age_sex_population,
        ).set_index("location_id")
        assert got.loc[10, "dengue_inc_count"] == pytest.approx(16.0)
        assert got.loc[1, "dengue_inc_rate"] == pytest.approx(16.0 / 300.0)


class TestSummarizeProducts:
    def test_collapses_draws_to_mean_lower_upper(self):
        rows = [
            {"location_id": 1, "year_id": YEAR, "draw": d,
             "dengue_inc_rate": 0.1 * d, "dengue_inc_count": 10.0 * d,
             "dengue_mort_rate": 0.0, "dengue_mort_count": 0.0}
            for d in range(1, 101)
        ]
        got = summarize_products(pd.DataFrame(rows))
        assert got["dengue_inc_count_mean"].iloc[0] == pytest.approx(505.0)
        assert got["dengue_inc_count_lower"].iloc[0] < got["dengue_inc_count_mean"].iloc[0]
        assert got["dengue_inc_count_upper"].iloc[0] > got["dengue_inc_count_mean"].iloc[0]


class TestBuildHierarchyProducts:
    def test_without_draws_returns_a_point_estimate(
        self, leaf_rates, population, hierarchy,
    ):
        """The in-sample case: past covariates are single-realization."""
        got = build_hierarchy_products(leaf_rates, population, hierarchy)
        assert "dengue_inc_rate" in got.columns
        assert "dengue_inc_rate_mean" not in got.columns
        assert set(got["location_id"]) == {1, 10, 20}

    def test_with_draws_returns_mean_lower_upper(self, population, hierarchy):
        rows = []
        for draw in range(20):
            for loc, rate in ((10, 0.10), (20, 0.20)):
                rows.append({"location_id": loc, "year_id": YEAR, "draw": draw,
                             "dengue_inc_rate": rate * (1 + 0.01 * draw),
                             "dengue_mort_rate": 0.0})
        got = build_hierarchy_products(
            pd.DataFrame(rows), population, hierarchy, draw_column="draw")
        assert "dengue_inc_rate_mean" in got.columns
        root = got[got["location_id"] == 1].iloc[0]
        assert root["dengue_inc_rate_mean"] > 0

    def test_rate_summary_equals_count_summary_over_population(
        self, population, hierarchy,
    ):
        """Population has no draw axis, so summarise-then-divide is exact."""
        rows = []
        for draw in range(50):
            for loc, rate in ((10, 0.10), (20, 0.20)):
                rows.append({"location_id": loc, "year_id": YEAR, "draw": draw,
                             "dengue_inc_rate": rate * (1 + 0.02 * draw),
                             "dengue_mort_rate": 0.0})
        got = build_hierarchy_products(
            pd.DataFrame(rows), population, hierarchy, draw_column="draw")
        root = got[got["location_id"] == 1].iloc[0]
        assert root["dengue_inc_rate_mean"] == pytest.approx(
            root["dengue_inc_count_mean"] / 300.0)
