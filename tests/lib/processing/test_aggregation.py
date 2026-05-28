"""
Tests for lib/processing/aggregation.py
"""

import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing.aggregation import (
    aggregate_aa_count_lsae_to_gbd,
    aggregate_aa_rate_lsae_to_gbd,
    aggregate_level,
    aggregate_to_parent,
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
