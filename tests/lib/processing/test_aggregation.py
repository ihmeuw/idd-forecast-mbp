"""
Tests for lib/processing/aggregation.py
"""

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.aggregation import (
    aggregate_aa_count_lsae_to_gbd,
    aggregate_outcomes_to_ancestors,
    aggregate_aa_rate_lsae_to_gbd,
    aggregate_level,
    aggregate_to_parent,
    make_rate_from_count,
    roll_up_covariates_to_ancestors,
    roll_up_hierarchy,
    roll_up_to_ancestors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_hierarchy():
    """Minimal 3-level hierarchy: 1 global → 2 regions → 4 level-3 → 8 level-4 → 16 level-5."""
    rows = []
    # Level 0: global
    rows.append({'location_id': 1, 'parent_id': 0, 'level': 0})
    # Level 1: 2 super-regions
    for i in range(2, 4):
        rows.append({'location_id': i, 'parent_id': 1, 'level': 1})
    # Level 2: 4 regions (2 per super-region)
    for i in range(4, 8):
        rows.append({'location_id': i, 'parent_id': (i // 2), 'level': 2})
    # Level 3: 8 countries
    for i in range(8, 16):
        rows.append({'location_id': i, 'parent_id': (i // 2), 'level': 3})
    # Level 4: 16 locs
    for i in range(16, 32):
        rows.append({'location_id': i, 'parent_id': (i // 2), 'level': 4})
    # Level 5: 32 locs
    for i in range(32, 64):
        rows.append({'location_id': i, 'parent_id': (i // 2), 'level': 5})
    return pd.DataFrame(rows)


@pytest.fixture
def level5_count_df(simple_hierarchy):
    """Level-5 counts: 1 for each location, 2 years."""
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()
    rows = []
    for loc in level5_ids:
        for year in [2020, 2021]:
            rows.append({'location_id': loc, 'year_id': year, 'count': 1.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# aggregate_level
# ---------------------------------------------------------------------------

def test_aggregate_level_sums_children(simple_hierarchy, level5_count_df):
    # Each level-5 parent has 2 children, each with count=1 → parent sum = 2
    result = aggregate_level('count', level5_count_df, simple_hierarchy)
    assert 'location_id' in result.columns
    # All level-4 parents should have count = 2
    assert (result['count'] == 2.0).all()


def test_aggregate_level_output_columns(simple_hierarchy, level5_count_df):
    result = aggregate_level('count', level5_count_df, simple_hierarchy)
    assert set(result.columns) == {'location_id', 'year_id', 'count'}


def test_aggregate_level_years_preserved(simple_hierarchy, level5_count_df):
    result = aggregate_level('count', level5_count_df, simple_hierarchy)
    assert set(result['year_id'].unique()) == {2020, 2021}


# ---------------------------------------------------------------------------
# aggregate_to_parent
# ---------------------------------------------------------------------------

def test_aggregate_to_parent_aa(simple_hierarchy, level5_count_df):
    result = aggregate_to_parent(level5_count_df, simple_hierarchy, 'count')
    assert 'location_id' in result.columns
    assert 'parent_id' not in result.columns
    assert (result['count'] == 2.0).all()


def test_aggregate_to_parent_as_preserves_age_sex(simple_hierarchy):
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()[:4]
    rows = []
    for loc in level5_ids:
        for age in [10, 15]:
            for sex in [1, 2]:
                rows.append({
                    'location_id': loc, 'year_id': 2020,
                    'age_group_id': age, 'sex_id': sex, 'count': 1.0,
                })
    df = pd.DataFrame(rows)

    result = aggregate_to_parent(df, simple_hierarchy, 'count', preserve_age_sex=True)
    assert 'age_group_id' in result.columns
    assert 'sex_id' in result.columns


def test_aggregate_to_parent_aa_drops_age_sex(simple_hierarchy, level5_count_df):
    # AA mode should not have age_group_id / sex_id in output (they're not in input either)
    result = aggregate_to_parent(level5_count_df, simple_hierarchy, 'count')
    assert 'age_group_id' not in result.columns
    assert 'sex_id' not in result.columns


def test_aggregate_to_parent_extra_group_cols_preserves_draw(simple_hierarchy):
    """extra_group_cols=['draw'] keeps the draw dimension and sums within each draw."""
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()
    rows = []
    for loc in level5_ids:
        for draw in [0, 1]:
            rows.append({'location_id': loc, 'year_id': 2020, 'draw': draw,
                         'count': 1.0 + draw})  # draw 0 -> 1.0, draw 1 -> 2.0
    df = pd.DataFrame(rows)

    result = aggregate_to_parent(df, simple_hierarchy, 'count', extra_group_cols=['draw'])
    assert 'draw' in result.columns
    # each level-4 parent has 2 children; draw 0 -> 2.0, draw 1 -> 4.0
    assert (result[result['draw'] == 0]['count'] == 2.0).all()
    assert (result[result['draw'] == 1]['count'] == 4.0).all()


# ---------------------------------------------------------------------------
# roll_up_hierarchy
# ---------------------------------------------------------------------------

def test_roll_up_hierarchy_covers_all_levels(simple_hierarchy, level5_count_df):
    result = roll_up_hierarchy(level5_count_df, simple_hierarchy, 'count')
    got = set(result['location_id'].unique())
    want = set(simple_hierarchy['location_id'].unique())
    assert want.issubset(got)


def test_roll_up_hierarchy_global_is_sum_of_all_leaves(simple_hierarchy, level5_count_df):
    # 32 level-5 leaves × count 1 → global (location_id 1) should be 32 per year.
    result = roll_up_hierarchy(level5_count_df, simple_hierarchy, 'count')
    global_2020 = result[(result['location_id'] == 1) & (result['year_id'] == 2020)]['count']
    assert global_2020.iloc[0] == 32.0


def test_roll_up_hierarchy_draw_aware(simple_hierarchy):
    """With draws preserved, each draw rolls up independently to global."""
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()
    rows = []
    for loc in level5_ids:
        for draw in [0, 1]:
            rows.append({'location_id': loc, 'year_id': 2020, 'draw': draw,
                         'count': 1.0 + draw})
    df = pd.DataFrame(rows)

    result = roll_up_hierarchy(df, simple_hierarchy, 'count', extra_group_cols=['draw'])
    g = result[(result['location_id'] == 1) & (result['year_id'] == 2020)]
    # 32 leaves: draw 0 -> 32×1 = 32, draw 1 -> 32×2 = 64
    assert g[g['draw'] == 0]['count'].iloc[0] == 32.0
    assert g[g['draw'] == 1]['count'].iloc[0] == 64.0


def test_roll_up_hierarchy_preserves_age_sex(simple_hierarchy):
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()
    rows = []
    for loc in level5_ids:
        for age in [10, 15]:
            for sex in [1, 2]:
                rows.append({'location_id': loc, 'year_id': 2020,
                             'age_group_id': age, 'sex_id': sex, 'count': 1.0})
    df = pd.DataFrame(rows)

    result = roll_up_hierarchy(df, simple_hierarchy, 'count', preserve_age_sex=True)
    assert {'age_group_id', 'sex_id'}.issubset(result.columns)
    # global, one age/sex stratum: 32 leaves × 1 = 32
    g = result[(result['location_id'] == 1) & (result['age_group_id'] == 10) & (result['sex_id'] == 1)]
    assert g['count'].iloc[0] == 32.0


# ---------------------------------------------------------------------------
# aggregate_aa_count_lsae_to_gbd
# ---------------------------------------------------------------------------

def test_aggregate_aa_count_returns_none_by_default(simple_hierarchy, level5_count_df):
    result = aggregate_aa_count_lsae_to_gbd(
        'count', simple_hierarchy, level5_count_df,
    )
    assert result is None


def test_aggregate_aa_count_return_full_df(simple_hierarchy, level5_count_df):
    result = aggregate_aa_count_lsae_to_gbd(
        'count', simple_hierarchy, level5_count_df, return_full_df=True,
    )
    assert isinstance(result, pd.DataFrame)


def test_aggregate_aa_count_covers_all_levels(simple_hierarchy, level5_count_df):
    result = aggregate_aa_count_lsae_to_gbd(
        'count', simple_hierarchy, level5_count_df, return_full_df=True,
    )
    # All hierarchy locations should appear in the result
    result_locs = set(result['location_id'].unique())
    hierarchy_locs = set(simple_hierarchy['location_id'].unique())
    assert hierarchy_locs.issubset(result_locs)


def test_aggregate_aa_count_writes_parquet(tmp_path, simple_hierarchy, level5_count_df):
    out_path = tmp_path / 'out.parquet'
    aggregate_aa_count_lsae_to_gbd(
        'count', simple_hierarchy, level5_count_df,
        aa_full_count_df_path=str(out_path),
    )
    assert out_path.exists()


# ---------------------------------------------------------------------------
# aggregate_aa_rate_lsae_to_gbd
# ---------------------------------------------------------------------------

def test_aggregate_aa_rate_returns_none_by_default(simple_hierarchy, level5_count_df):
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()
    rows = []
    for loc in level5_ids:
        for year in [2020, 2021]:
            rows.append({'location_id': loc, 'year_id': year, 'rate': 0.001})
    rate_df = pd.DataFrame(rows)

    pop_rows = []
    for loc in simple_hierarchy['location_id'].tolist():
        for year in [2020, 2021]:
            pop_rows.append({'location_id': loc, 'year_id': year, 'population': 1000.0})
    pop_df = pd.DataFrame(pop_rows)

    result = aggregate_aa_rate_lsae_to_gbd(
        'rate', simple_hierarchy, rate_df, pop_df,
    )
    assert result is None


def test_aggregate_aa_rate_return_full_df(simple_hierarchy):
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()
    rows = []
    for loc in level5_ids:
        for year in [2020, 2021]:
            rows.append({'location_id': loc, 'year_id': year, 'rate': 0.001})
    rate_df = pd.DataFrame(rows)

    pop_rows = []
    for loc in simple_hierarchy['location_id'].tolist():
        for year in [2020, 2021]:
            pop_rows.append({'location_id': loc, 'year_id': year, 'population': 1000.0})
    pop_df = pd.DataFrame(pop_rows)

    result = aggregate_aa_rate_lsae_to_gbd(
        'rate', simple_hierarchy, rate_df, pop_df, return_full_df=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert 'rate' in result.columns
    assert 'tmp_count' not in result.columns


def test_aggregate_aa_rate_drops_population_from_rate_df(simple_hierarchy):
    """rate_df with a 'population' column: should be dropped before processing."""
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()
    rows = []
    for loc in level5_ids:
        for year in [2020, 2021]:
            rows.append({'location_id': loc, 'year_id': year, 'rate': 0.001, 'population': 500.0})
    rate_df = pd.DataFrame(rows)

    pop_rows = []
    for loc in simple_hierarchy['location_id'].tolist():
        for year in [2020, 2021]:
            pop_rows.append({'location_id': loc, 'year_id': year, 'population': 1000.0})
    pop_df = pd.DataFrame(pop_rows)

    result = aggregate_aa_rate_lsae_to_gbd(
        'rate', simple_hierarchy, rate_df, pop_df, return_full_df=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert 'rate' in result.columns


def test_aggregate_aa_rate_no_level_in_output(simple_hierarchy):
    """'level' column should be dropped from the final rate result."""
    level5_ids = simple_hierarchy[simple_hierarchy['level'] == 5]['location_id'].tolist()
    rows = [{'location_id': loc, 'year_id': 2020, 'rate': 0.001} for loc in level5_ids]
    rate_df = pd.DataFrame(rows)

    pop_rows = [{'location_id': loc, 'year_id': 2020, 'population': 1000.0}
                for loc in simple_hierarchy['location_id'].tolist()]
    pop_df = pd.DataFrame(pop_rows)

    result = aggregate_aa_rate_lsae_to_gbd(
        'rate', simple_hierarchy, rate_df, pop_df, return_full_df=True,
    )
    assert 'level' not in result.columns


# ---------------------------------------------------------------------------
# make_rate_from_count
# ---------------------------------------------------------------------------

from idd_forecast_mbp.lib.processing.aggregation import make_rate_from_count


@pytest.fixture
def count_and_pop(simple_hierarchy):
    all_locs = simple_hierarchy['location_id'].tolist()
    count_rows = [{'location_id': loc, 'year_id': 2020, 'count': 10.0} for loc in all_locs]
    pop_rows = [{'location_id': loc, 'year_id': 2020, 'population': 1000.0} for loc in all_locs]
    return pd.DataFrame(count_rows), pd.DataFrame(pop_rows)


def test_make_rate_from_count_returns_none_by_default(count_and_pop):
    count_df, pop_df = count_and_pop
    result = make_rate_from_count('rate', 'count', count_df, pop_df)
    assert result is None


def test_make_rate_from_count_return_full_df(count_and_pop):
    count_df, pop_df = count_and_pop
    result = make_rate_from_count('rate', 'count', count_df, pop_df, return_full_df=True)
    assert isinstance(result, pd.DataFrame)
    assert 'rate' in result.columns
    assert 'count' not in result.columns


def test_make_rate_from_count_drops_population_column(count_and_pop):
    """count_df with a 'population' column should be dropped before the merge."""
    count_df, pop_df = count_and_pop
    count_df = count_df.copy()
    count_df['population'] = 999.0
    result = make_rate_from_count('rate', 'count', count_df, pop_df, return_full_df=True)
    assert isinstance(result, pd.DataFrame)
    assert 'rate' in result.columns


def test_make_rate_from_count_writes_parquet(tmp_path, count_and_pop):
    count_df, pop_df = count_and_pop
    out = tmp_path / 'rate.parquet'
    make_rate_from_count('rate', 'count', count_df, pop_df, aa_full_rate_df_path=str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# roll_up_to_ancestors — mixed-level leaf sets (the FHS shape)
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_level_hierarchy():
    """Miniature of the FHS shape: leaves at BOTH level 3 and level 4.

    global 1 -> super-region 2 -> region 4 -> countries 8, 9.
    Country 8 has no subnationals, so it is itself a leaf (level 3).
    Country 9 has subnationals 16 and 17, which are the leaves (level 4).
    Leaf set = {8, 16, 17}, spanning two levels — exactly why the
    level-iterating roll-up is not usable here.
    """
    rows = [
        {'location_id': 1, 'parent_id': 0, 'level': 0, 'path_to_top_parent': '1'},
        {'location_id': 2, 'parent_id': 1, 'level': 1, 'path_to_top_parent': '1,2'},
        {'location_id': 4, 'parent_id': 2, 'level': 2, 'path_to_top_parent': '1,2,4'},
        {'location_id': 8, 'parent_id': 4, 'level': 3, 'path_to_top_parent': '1,2,4,8'},
        {'location_id': 9, 'parent_id': 4, 'level': 3, 'path_to_top_parent': '1,2,4,9'},
        {'location_id': 16, 'parent_id': 9, 'level': 4, 'path_to_top_parent': '1,2,4,9,16'},
        {'location_id': 17, 'parent_id': 9, 'level': 4, 'path_to_top_parent': '1,2,4,9,17'},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mixed_level_counts():
    """Counts at the three leaves, one year. Total = 18."""
    return pd.DataFrame([
        {'location_id': 8, 'year_id': 2023, 'count': 10.0},
        {'location_id': 16, 'year_id': 2023, 'count': 3.0},
        {'location_id': 17, 'year_id': 2023, 'count': 5.0},
    ])


def _by_loc(df, col='count'):
    return dict(zip(df['location_id'], df[col], strict=True))


def test_roll_up_to_ancestors_sums_into_every_ancestor(
    mixed_level_hierarchy, mixed_level_counts,
):
    got = _by_loc(roll_up_to_ancestors(mixed_level_counts, mixed_level_hierarchy, 'count'))
    assert got == {1: 18.0, 2: 18.0, 4: 18.0, 8: 10.0, 9: 8.0, 16: 3.0, 17: 5.0}


def test_roll_up_to_ancestors_keeps_the_level_3_leaf(
    mixed_level_hierarchy, mixed_level_counts,
):
    """The whole point: a level-3 leaf must survive and reach the global total.

    The level-iterating roll-up seeded at the finest level cannot see location 8,
    so its 10 never reaches the root. This asserts the contrast directly.
    """
    ancestors = roll_up_to_ancestors(mixed_level_counts, mixed_level_hierarchy, 'count')
    assert _by_loc(ancestors)[1] == 18.0

    by_level = roll_up_hierarchy(
        mixed_level_counts, mixed_level_hierarchy, 'count', start_level=4,
    )
    global_by_level = by_level[by_level['location_id'] == 1]['count'].sum()
    assert global_by_level == 8.0, "level iteration is expected to lose the level-3 leaf"


def test_roll_up_to_ancestors_multiple_count_columns(mixed_level_hierarchy):
    df = pd.DataFrame([
        {'location_id': 8, 'year_id': 2023, 'inc': 10.0, 'mort': 1.0},
        {'location_id': 16, 'year_id': 2023, 'inc': 3.0, 'mort': 2.0},
        {'location_id': 17, 'year_id': 2023, 'inc': 5.0, 'mort': 4.0},
    ])
    out = roll_up_to_ancestors(df, mixed_level_hierarchy, ['inc', 'mort'])
    root = out[out['location_id'] == 1].iloc[0]
    assert root['inc'] == 18.0
    assert root['mort'] == 7.0


def test_roll_up_to_ancestors_is_draw_aware(mixed_level_hierarchy):
    """Aggregation happens within a draw; draws are never mixed."""
    rows = []
    for draw, scale in ((0, 1.0), (1, 10.0)):
        for loc, val in ((8, 10.0), (16, 3.0), (17, 5.0)):
            rows.append({'location_id': loc, 'year_id': 2023, 'draw': draw,
                         'count': val * scale})
    out = roll_up_to_ancestors(pd.DataFrame(rows), mixed_level_hierarchy, 'count',
                               extra_group_cols=['draw'])
    root = out[out['location_id'] == 1].set_index('draw')['count']
    assert root.loc[0] == 18.0
    assert root.loc[1] == 180.0


def test_roll_up_to_ancestors_preserves_age_sex(mixed_level_hierarchy):
    rows = [
        {'location_id': loc, 'year_id': 2023, 'age_group_id': age, 'sex_id': sex,
         'count': 1.0}
        for loc in (8, 16, 17) for age in (3, 4) for sex in (1, 2)
    ]
    out = roll_up_to_ancestors(pd.DataFrame(rows), mixed_level_hierarchy, 'count',
                               preserve_age_sex=True)
    root = out[out['location_id'] == 1]
    assert len(root) == 4  # 2 ages x 2 sexes, not collapsed
    assert set(root['count']) == {3.0}  # three leaves per cell


def test_roll_up_to_ancestors_preserves_years(mixed_level_hierarchy):
    rows = [
        {'location_id': loc, 'year_id': yr, 'count': 1.0}
        for loc in (8, 16, 17) for yr in (2023, 2024)
    ]
    out = roll_up_to_ancestors(pd.DataFrame(rows), mixed_level_hierarchy, 'count')
    root = out[out['location_id'] == 1].set_index('year_id')['count']
    assert root.loc[2023] == 3.0
    assert root.loc[2024] == 3.0


def test_roll_up_to_ancestors_returns_integer_location_ids(
    mixed_level_hierarchy, mixed_level_counts,
):
    out = roll_up_to_ancestors(mixed_level_counts, mixed_level_hierarchy, 'count')
    assert pd.api.types.is_integer_dtype(out['location_id'])


def test_roll_up_to_ancestors_rejects_unknown_location(mixed_level_hierarchy):
    df = pd.DataFrame([{'location_id': 999, 'year_id': 2023, 'count': 1.0}])
    with pytest.raises(KeyError, match='absent from the hierarchy'):
        roll_up_to_ancestors(df, mixed_level_hierarchy, 'count')


def test_roll_up_to_ancestors_requires_path_column(mixed_level_counts):
    bare = pd.DataFrame([{'location_id': 8, 'parent_id': 4, 'level': 3}])
    with pytest.raises(KeyError, match='path_to_top_parent'):
        roll_up_to_ancestors(mixed_level_counts, bare, 'count')


def test_roll_up_to_ancestors_tolerates_duplicate_hierarchy_rows(
    mixed_level_hierarchy, mixed_level_counts,
):
    """A hierarchy carrying repeated location rows must not double count."""
    doubled = pd.concat([mixed_level_hierarchy, mixed_level_hierarchy], ignore_index=True)
    got = _by_loc(roll_up_to_ancestors(mixed_level_counts, doubled, 'count'))
    assert got[1] == 18.0


# ---------------------------------------------------------------------------
# roll_up_covariates_to_ancestors + make_rate_from_count(join_cols=...)
# ---------------------------------------------------------------------------

class TestRollUpCovariatesToAncestors:
    """Population-weighted covariate aggregation into every ancestor.

    Fixture shape mirrors the real motivating case: two leaves at DIFFERENT levels
    under one region, which is what breaks a level-by-level roll-up.
    """

    @staticmethod
    def _hierarchy():
        return pd.DataFrame({
            "location_id":        [1, 100, 10, 11, 20],
            "level":              [0,   2,  3,  4,  3],
            "path_to_top_parent": ["1", "1,100", "1,100,10", "1,100,10,11", "1,100,20"],
        })

    @staticmethod
    def _cov():
        # leaf 11 (pop 300) and leaf 20 (pop 100) are the leaves we aggregate
        return pd.DataFrame({
            "location_id": [11, 20],
            "year_id":     [2000, 2000],
            "gdppc_mean":  [10.0, 50.0],
        })

    @staticmethod
    def _pop():
        return pd.DataFrame({
            "location_id": [11, 20],
            "year_id":     [2000, 2000],
            "population":  [300.0, 100.0],
        })

    def test_weighted_not_plain_mean(self):
        out = roll_up_covariates_to_ancestors(
            self._cov(), self._hierarchy(), "gdppc_mean", self._pop())
        region = out.loc[out.location_id == 100, "gdppc_mean"].iloc[0]
        # (10*300 + 50*100) / 400 = 20.0 ; a plain mean would be 30.0
        assert region == pytest.approx(20.0)
        assert region != pytest.approx(30.0)

    def test_leaves_pass_through_unchanged(self):
        out = roll_up_covariates_to_ancestors(
            self._cov(), self._hierarchy(), "gdppc_mean", self._pop())
        assert out.loc[out.location_id == 11, "gdppc_mean"].iloc[0] == pytest.approx(10.0)
        assert out.loc[out.location_id == 20, "gdppc_mean"].iloc[0] == pytest.approx(50.0)

    def test_mixed_level_leaves_both_reach_the_ancestor(self):
        """Leaf 11 is level 4 and leaf 20 is level 3; both must count."""
        out = roll_up_covariates_to_ancestors(
            self._cov(), self._hierarchy(), "gdppc_mean", self._pop())
        # global sees both leaves, so it equals the region here
        assert out.loc[out.location_id == 1, "gdppc_mean"].iloc[0] == pytest.approx(20.0)
        assert set(out.location_id) == {1, 100, 10, 11, 20}

    def test_intermediate_ancestor_sees_only_its_own_descendants(self):
        out = roll_up_covariates_to_ancestors(
            self._cov(), self._hierarchy(), "gdppc_mean", self._pop())
        # location 10's only leaf is 11
        assert out.loc[out.location_id == 10, "gdppc_mean"].iloc[0] == pytest.approx(10.0)

    def test_multiple_covariates_at_once(self):
        cov = self._cov().assign(temp=[25.0, 30.0])
        out = roll_up_covariates_to_ancestors(
            cov, self._hierarchy(), ["gdppc_mean", "temp"], self._pop())
        row = out[out.location_id == 100].iloc[0]
        assert row["gdppc_mean"] == pytest.approx(20.0)
        assert row["temp"] == pytest.approx((25.0 * 300 + 30.0 * 100) / 400)

    def test_age_sex_duplication_is_collapsed_before_weighting(self):
        """Covariates are location-year; duplicated cells must not double-weight."""
        dup = pd.concat([self._cov()] * 4, ignore_index=True)
        out = roll_up_covariates_to_ancestors(
            dup, self._hierarchy(), "gdppc_mean", self._pop())
        assert out.loc[out.location_id == 100, "gdppc_mean"].iloc[0] == pytest.approx(20.0)

    def test_nan_covariate_propagates_rather_than_being_skipped(self):
        cov = self._cov().copy()
        cov.loc[cov.location_id == 20, "gdppc_mean"] = np.nan
        out = roll_up_covariates_to_ancestors(
            cov, self._hierarchy(), "gdppc_mean", self._pop())
        assert np.isnan(out.loc[out.location_id == 100, "gdppc_mean"].iloc[0])
        # the unaffected branch is still fine
        assert out.loc[out.location_id == 10, "gdppc_mean"].iloc[0] == pytest.approx(10.0)

    def test_missing_population_raises(self):
        pop = self._pop()
        pop = pop[pop.location_id != 20]
        with pytest.raises(ValueError, match="no population"):
            roll_up_covariates_to_ancestors(
                self._cov(), self._hierarchy(), "gdppc_mean", pop)

    def test_missing_path_column_raises(self):
        h = self._hierarchy().drop(columns=["path_to_top_parent"])
        with pytest.raises(KeyError, match="path_to_top_parent"):
            roll_up_covariates_to_ancestors(
                self._cov(), h, "gdppc_mean", self._pop())

    def test_location_absent_from_hierarchy_raises(self):
        cov = pd.concat([self._cov(),
                         pd.DataFrame({"location_id": [999], "year_id": [2000],
                                       "gdppc_mean": [1.0]})], ignore_index=True)
        pop = pd.concat([self._pop(),
                         pd.DataFrame({"location_id": [999], "year_id": [2000],
                                       "population": [1.0]})], ignore_index=True)
        with pytest.raises(KeyError, match="absent from the hierarchy"):
            roll_up_covariates_to_ancestors(
                cov, self._hierarchy(), "gdppc_mean", pop)


class TestMakeRateFromCountJoinCols:
    """The age/sex grain must divide by the age/sex population, not the all-age one."""

    @staticmethod
    def _counts():
        return pd.DataFrame({
            "location_id":   [10, 10, 10, 10],
            "year_id":       [2000] * 4,
            "age_group_id":  [2, 2, 3, 3],
            "sex_id":        [1, 2, 1, 2],
            "inc_count":     [10.0, 20.0, 30.0, 40.0],
        })

    @staticmethod
    def _as_pop():
        return pd.DataFrame({
            "location_id":   [10, 10, 10, 10],
            "year_id":       [2000] * 4,
            "age_group_id":  [2, 2, 3, 3],
            "sex_id":        [1, 2, 1, 2],
            "population":    [100.0, 200.0, 300.0, 400.0],
        })

    def test_age_sex_join_divides_by_the_matching_cell(self):
        out = make_rate_from_count(
            "inc_rate", "inc_count", self._counts(), self._as_pop(),
            return_full_df=True,
            join_cols=("location_id", "year_id", "age_group_id", "sex_id"))
        assert out["inc_rate"].tolist() == pytest.approx([0.1, 0.1, 0.1, 0.1])

    def test_default_join_is_unchanged_all_age_behaviour(self):
        counts = pd.DataFrame({"location_id": [10], "year_id": [2000], "inc_count": [50.0]})
        pop = pd.DataFrame({"location_id": [10], "year_id": [2000], "population": [1000.0]})
        out = make_rate_from_count(
            "inc_rate", "inc_count", counts, pop, return_full_df=True)
        assert out["inc_rate"].iloc[0] == pytest.approx(0.05)

    def test_all_age_population_on_age_sex_counts_is_the_bug_join_cols_prevents(self):
        """Guard: the default join broadcasts one population across every cell."""
        aa_pop = pd.DataFrame({"location_id": [10], "year_id": [2000], "population": [1000.0]})
        wrong = make_rate_from_count(
            "inc_rate", "inc_count", self._counts(), aa_pop, return_full_df=True)
        assert wrong["inc_rate"].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04])
        right = make_rate_from_count(
            "inc_rate", "inc_count", self._counts(), self._as_pop(),
            return_full_df=True,
            join_cols=("location_id", "year_id", "age_group_id", "sex_id"))
        assert right["inc_rate"].tolist() != pytest.approx(wrong["inc_rate"].tolist())

    def test_zero_population_gives_zero_rate(self):
        pop = self._as_pop().copy()
        pop.loc[0, "population"] = 0.0
        out = make_rate_from_count(
            "inc_rate", "inc_count", self._counts(), pop, return_full_df=True,
            join_cols=("location_id", "year_id", "age_group_id", "sex_id"))
        assert out["inc_rate"].iloc[0] == 0.0


class TestAggregateOutcomesToAncestors:
    """Counts sum; rates come from the ancestor's OWN population, never the leaves' sum."""

    @staticmethod
    def _hierarchy():
        return pd.DataFrame({
            "location_id":        [1, 100, 11, 20],
            "level":              [0,   2,  4,  3],
            "path_to_top_parent": ["1", "1,100", "1,100,11", "1,100,20"],
        })

    @staticmethod
    def _leaves():
        return pd.DataFrame({
            "location_id": [11, 20],
            "year_id":     [2000, 2000],
            "inc_count":   [10.0, 30.0],
        })

    @staticmethod
    def _aa_pop():
        """Region 100's own population (1000) EXCEEDS its leaves' sum (400)."""
        return pd.DataFrame({
            "location_id": [1, 100, 11, 20],
            "year_id":     [2000] * 4,
            "population":  [2000.0, 1000.0, 300.0, 100.0],
        })

    def test_counts_sum_into_the_ancestor(self):
        out = aggregate_outcomes_to_ancestors(
            self._leaves(), self._hierarchy(), {"inc_count": "inc_rate"}, self._aa_pop())
        assert out.loc[out.location_id == 100, "inc_count"].iloc[0] == pytest.approx(40.0)

    def test_rate_uses_own_population_not_the_leaf_sum(self):
        """The regression this function exists to prevent."""
        out = aggregate_outcomes_to_ancestors(
            self._leaves(), self._hierarchy(), {"inc_count": "inc_rate"}, self._aa_pop())
        got = out.loc[out.location_id == 100, "inc_rate"].iloc[0]
        assert got == pytest.approx(40.0 / 1000.0)        # own population
        assert got != pytest.approx(40.0 / 400.0)         # summed-children denominator

    def test_leaf_rows_survive_and_use_their_own_population(self):
        out = aggregate_outcomes_to_ancestors(
            self._leaves(), self._hierarchy(), {"inc_count": "inc_rate"}, self._aa_pop())
        assert out.loc[out.location_id == 11, "inc_rate"].iloc[0] == pytest.approx(10.0 / 300.0)

    def test_multiple_outcomes(self):
        leaves = self._leaves().assign(mort_count=[1.0, 3.0])
        out = aggregate_outcomes_to_ancestors(
            leaves, self._hierarchy(),
            {"inc_count": "inc_rate", "mort_count": "mort_rate"}, self._aa_pop())
        row = out[out.location_id == 100].iloc[0]
        assert row["inc_rate"] == pytest.approx(40.0 / 1000.0)
        assert row["mort_rate"] == pytest.approx(4.0 / 1000.0)
        assert row["inc_count"] == pytest.approx(40.0)
        assert row["mort_count"] == pytest.approx(4.0)

    def test_preserve_age_sex_matches_population_per_cell(self):
        leaves = pd.DataFrame({
            "location_id":  [11, 11, 20, 20],
            "year_id":      [2000] * 4,
            "age_group_id": [2, 3, 2, 3],
            "sex_id":       [1, 1, 1, 1],
            "inc_count":    [1.0, 2.0, 3.0, 4.0],
        })
        as_pop = pd.DataFrame({
            "location_id":  [100, 100, 11, 11, 20, 20],
            "year_id":      [2000] * 6,
            "age_group_id": [2, 3, 2, 3, 2, 3],
            "sex_id":       [1] * 6,
            "population":   [400.0, 800.0, 100.0, 200.0, 50.0, 100.0],
        })
        out = aggregate_outcomes_to_ancestors(
            leaves, self._hierarchy(), {"inc_count": "inc_rate"}, as_pop,
            preserve_age_sex=True)
        a2 = out[(out.location_id == 100) & (out.age_group_id == 2)].iloc[0]
        a3 = out[(out.location_id == 100) & (out.age_group_id == 3)].iloc[0]
        assert a2["inc_count"] == pytest.approx(4.0)          # 1 + 3
        assert a2["inc_rate"] == pytest.approx(4.0 / 400.0)   # age-2 population
        assert a3["inc_count"] == pytest.approx(6.0)          # 2 + 4
        assert a3["inc_rate"] == pytest.approx(6.0 / 800.0)   # age-3 population

    def test_age_sex_rates_differ_from_using_all_age_population(self):
        """Guard against the all-age denominator leaking into an age/sex call."""
        leaves = pd.DataFrame({
            "location_id":  [11, 11], "year_id": [2000, 2000],
            "age_group_id": [2, 3], "sex_id": [1, 1], "inc_count": [1.0, 2.0],
        })
        as_pop = pd.DataFrame({
            "location_id":  [11, 11, 100, 100], "year_id": [2000] * 4,
            "age_group_id": [2, 3, 2, 3], "sex_id": [1] * 4,
            "population":   [100.0, 200.0, 100.0, 200.0],
        })
        out = aggregate_outcomes_to_ancestors(
            leaves, self._hierarchy(), {"inc_count": "inc_rate"}, as_pop,
            preserve_age_sex=True)
        rates = out[out.location_id == 11].sort_values("age_group_id")["inc_rate"].tolist()
        assert rates == pytest.approx([0.01, 0.01])

    def test_population_column_is_returned_for_audit(self):
        out = aggregate_outcomes_to_ancestors(
            self._leaves(), self._hierarchy(), {"inc_count": "inc_rate"}, self._aa_pop())
        assert "population" in out.columns
        assert out.loc[out.location_id == 100, "population"].iloc[0] == pytest.approx(1000.0)
