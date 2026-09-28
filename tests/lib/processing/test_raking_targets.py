"""Raking children to external parent targets, applying reference factors, and the checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idd_forecast_mbp.lib.processing import raking as rk

KEYS = ["year_id", "age_group_id", "sex_id"]
GROUP = ["parent_id", *KEYS]


def _children() -> pd.DataFrame:
    """Two parents; parent 10 has children 101, 102; parent 20 has 201, 202, 203. Two years x two ages x one sex."""
    rows = []
    for year in (2023, 2024):
        for age in (2, 3):
            for parent, kids in ((10, (101, 102)), (20, (201, 202, 203))):
                for i, kid in enumerate(kids, start=1):
                    rows.append(
                        {
                            "location_id": kid,
                            "parent_id": parent,
                            "year_id": year,
                            "age_group_id": age,
                            "sex_id": 1,
                            "value": float(i * 10 + (year - 2023) + age),
                        }
                    )
    return pd.DataFrame(rows)


def _targets(children: pd.DataFrame, scale: float = 2.0) -> pd.DataFrame:
    """Targets = ``scale`` x the children's sums, so the expected factor is ``scale`` everywhere."""
    sums = children.groupby(GROUP, as_index=False)["value"].sum()
    sums["value"] = sums["value"] * scale
    return sums


# ------------------------------------------------------------------ rake
def test_factor_is_target_over_children_sum_and_sums_match_after_raking() -> None:
    children = _children()
    targets = _targets(children, scale=2.0)
    res = rk.rake_children_to_parent_targets(
        children, targets, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    assert np.allclose(res.factors["factor"], 2.0)
    assert (res.factors["reason"] == rk.REASON_RAKED).all()
    assert res.excluded.empty
    after = res.raked.groupby(GROUP, as_index=False)["value"].sum()
    merged = after.merge(targets, on=GROUP, suffixes=("_raked", "_target"))
    assert np.allclose(merged["value_raked"], merged["value_target"])
    # every child of one parent-cell gets the same factor; row order and columns preserved
    assert list(res.raked.columns) == list(children.columns)
    assert len(res.raked) == len(children)


def test_factors_differ_by_cell() -> None:
    children = _children()
    targets = _targets(children, scale=1.0)
    targets.loc[(targets.parent_id == 10) & (targets.year_id == 2024), "value"] *= 3.0
    res = rk.rake_children_to_parent_targets(
        children, targets, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    f = res.factors.set_index(GROUP)["factor"]
    assert f.loc[(10, 2024, 2, 1)] == pytest.approx(3.0)
    assert f.loc[(10, 2023, 2, 1)] == pytest.approx(1.0)
    assert f.loc[(20, 2024, 2, 1)] == pytest.approx(1.0)


def test_target_column_may_have_another_name() -> None:
    children = _children()
    targets = _targets(children).rename(columns={"value": "fhs"})
    res = rk.rake_children_to_parent_targets(
        children,
        targets,
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
        target_col="fhs",
    )
    assert np.allclose(res.factors["factor"], 2.0)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        ("target_zero", rk.REASON_TARGET_ZERO),
        ("children_zero", rk.REASON_CHILDREN_ZERO),
        ("no_target", rk.REASON_NO_TARGET),
    ],
)
def test_zero_rule_leaves_the_cell_unraked_and_reports_it(
    mutate: str, reason: str
) -> None:
    children = _children()
    targets = _targets(children, scale=2.0)
    cell = (
        (targets.parent_id == 20)
        & (targets.year_id == 2023)
        & (targets.age_group_id == 3)
    )
    if mutate == "target_zero":
        targets.loc[cell, "value"] = 0.0
    elif mutate == "children_zero":
        kids = (
            (children.parent_id == 20)
            & (children.year_id == 2023)
            & (children.age_group_id == 3)
        )
        children.loc[kids, "value"] = 0.0
    else:
        targets = targets[~cell]
    res = rk.rake_children_to_parent_targets(
        children, targets, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    assert len(res.excluded) == 1
    ex = res.excluded.iloc[0]
    assert (ex["parent_id"], ex["year_id"], ex["age_group_id"], ex["reason"]) == (
        20,
        2023,
        3,
        reason,
    )
    assert ex["factor"] == 1.0
    kids = (
        (children.parent_id == 20)
        & (children.year_id == 2023)
        & (children.age_group_id == 3)
    )
    before = children.loc[kids].sort_values("location_id")["value"].to_numpy()
    after = res.raked.loc[kids].sort_values("location_id")["value"].to_numpy()
    assert np.array_equal(before, after)  # left as they were
    # every other cell raked by 2
    others = res.factors[res.factors["reason"] == rk.REASON_RAKED]
    assert np.allclose(others["factor"], 2.0)


def test_children_zero_takes_precedence_over_target_zero() -> None:
    children = _children()
    targets = _targets(children)
    cell = (
        (targets.parent_id == 10)
        & (targets.year_id == 2023)
        & (targets.age_group_id == 2)
    )
    targets.loc[cell, "value"] = 0.0
    kids = (
        (children.parent_id == 10)
        & (children.year_id == 2023)
        & (children.age_group_id == 2)
    )
    children.loc[kids, "value"] = 0.0
    res = rk.rake_children_to_parent_targets(
        children, targets, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    assert res.excluded["reason"].tolist() == [rk.REASON_CHILDREN_ZERO]


def test_parents_in_targets_without_children_are_ignored() -> None:
    children = _children()
    targets = pd.concat(
        [
            _targets(children),
            pd.DataFrame(
                [
                    {
                        "parent_id": 99,
                        "year_id": 2023,
                        "age_group_id": 2,
                        "sex_id": 1,
                        "value": 5.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    res = rk.rake_children_to_parent_targets(
        children, targets, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    assert 99 not in set(res.factors["parent_id"])


@pytest.mark.parametrize(
    ("break_it", "match"),
    [
        ("zero_rule", "zero_rule must be one of"),
        ("missing_child_col", "children lacks"),
        ("missing_target_col", "targets lacks"),
        ("duplicate_targets", "more than one row"),
        ("negative_children", "negative"),
        ("negative_targets", "negative"),
    ],
)
def test_rake_refusals(break_it: str, match: str) -> None:
    children = _children()
    targets = _targets(children)
    kwargs: dict[str, object] = {
        "parent_col": "parent_id",
        "keys": KEYS,
        "value_col": "value",
    }
    if break_it == "zero_rule":
        kwargs["zero_rule"] = "zero"
    elif break_it == "missing_child_col":
        children = children.drop(columns="sex_id")
    elif break_it == "missing_target_col":
        targets = targets.drop(columns="value")
    elif break_it == "duplicate_targets":
        targets = pd.concat([targets, targets.head(1)], ignore_index=True)
    elif break_it == "negative_children":
        children.loc[0, "value"] = -1.0
    else:
        targets.loc[0, "value"] = -1.0
    with pytest.raises(ValueError, match=match):
        rk.rake_children_to_parent_targets(children, targets, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ apply to another arm
def test_reference_factors_scale_another_arm_cell_by_cell() -> None:
    reference = _children()
    targets = _targets(reference, scale=1.0)
    targets.loc[targets.parent_id == 20, "value"] *= 1.5
    res = rk.rake_children_to_parent_targets(
        reference, targets, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    arm = reference.copy()
    arm["value"] = arm["value"] * 0.7  # a hold arm: different values, same grid
    out = rk.apply_raking_factors(
        arm, res.factors, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    expected = np.where(arm["parent_id"] == 20, arm["value"] * 1.5, arm["value"])
    assert np.allclose(out["value"], expected)
    assert list(out.columns) == list(arm.columns)


def test_apply_refuses_a_cell_without_a_factor() -> None:
    reference = _children()
    res = rk.rake_children_to_parent_targets(
        reference,
        _targets(reference),
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
    )
    arm = reference.copy()
    arm.loc[0, "year_id"] = 2099
    with pytest.raises(ValueError, match="no raking factor"):
        rk.apply_raking_factors(
            arm, res.factors, parent_col="parent_id", keys=KEYS, value_col="value"
        )


def test_apply_refuses_duplicate_factors_and_missing_columns() -> None:
    children = _children()
    res = rk.rake_children_to_parent_targets(
        children,
        _targets(children),
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
    )
    dup = pd.concat([res.factors, res.factors.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="more than one row"):
        rk.apply_raking_factors(
            children, dup, parent_col="parent_id", keys=KEYS, value_col="value"
        )
    with pytest.raises(ValueError, match="factors lacks"):
        rk.apply_raking_factors(
            children,
            res.factors.drop(columns="factor"),
            parent_col="parent_id",
            keys=KEYS,
            value_col="value",
        )


# ------------------------------------------------------------------ checks
def test_check_passes_on_a_clean_rake_and_reports_counts() -> None:
    children = _children()
    targets = _targets(children)
    res = rk.rake_children_to_parent_targets(
        children, targets, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    rep = rk.check_raked_matches_targets(
        res.raked,
        targets,
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
        exclude=res.excluded,
        rel_tol=1e-9,
    )
    assert rep.ok
    assert rep.n_compared == len(targets)
    assert rep.n_excluded == 0
    assert rep.max_rel_diff <= 1e-12
    assert "OK" in rep.summary()
    rep.assert_ok()


def test_check_excludes_the_zero_rule_cells_so_it_can_pass() -> None:
    children = _children()
    targets = _targets(children)
    cell = (
        (targets.parent_id == 10)
        & (targets.year_id == 2024)
        & (targets.age_group_id == 2)
    )
    targets.loc[cell, "value"] = 0.0
    res = rk.rake_children_to_parent_targets(
        children, targets, parent_col="parent_id", keys=KEYS, value_col="value"
    )
    without = rk.check_raked_matches_targets(
        res.raked,
        targets,
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
        rel_tol=1e-9,
    )
    assert not without.ok
    assert np.isinf(without.max_rel_diff)  # children nonzero against a zero target
    with_ex = rk.check_raked_matches_targets(
        res.raked,
        targets,
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
        exclude=res.excluded,
        rel_tol=1e-9,
    )
    assert with_ex.ok
    assert with_ex.n_excluded == 1
    assert with_ex.n_compared == len(targets) - 1


def test_check_fails_beyond_tolerance_and_assert_ok_raises() -> None:
    children = _children()
    targets = _targets(children)
    rep = rk.check_raked_matches_targets(
        children,
        targets,
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
        rel_tol=0.1,
    )  # unraked children are half the targets
    assert not rep.ok
    assert rep.max_rel_diff == pytest.approx(0.5)
    with pytest.raises(ValueError, match="FAIL"):
        rep.assert_ok()


def test_check_with_nothing_compared_is_not_ok() -> None:
    children = _children()
    targets = _targets(children)
    targets["year_id"] += 100
    rep = rk.check_raked_matches_targets(
        children,
        targets,
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
        rel_tol=1.0,
    )
    assert rep.n_compared == 0
    assert not rep.ok


def test_check_target_column_name_and_missing_columns() -> None:
    children = _children()
    targets = _targets(children, scale=1.0).rename(columns={"value": "fhs"})
    rep = rk.check_raked_matches_targets(
        children,
        targets,
        parent_col="parent_id",
        keys=KEYS,
        value_col="value",
        target_col="fhs",
        rel_tol=1e-9,
    )
    assert rep.ok
    with pytest.raises(ValueError, match="raked lacks"):
        rk.check_raked_matches_targets(
            children.drop(columns="value"),
            targets,
            parent_col="parent_id",
            keys=KEYS,
            value_col="value",
            target_col="fhs",
            rel_tol=1e-9,
        )


def _measure(children: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = children[["location_id", *KEYS, "value"]].copy()
    out["value"] = out["value"] * scale
    return out


def test_sum_identity_holds_for_yll_plus_yld_and_fails_when_it_does_not() -> None:
    children = _children()
    yll, yld = _measure(children, 3.0), _measure(children, 0.5)
    daly = _measure(children, 3.5)
    keys = ["location_id", *KEYS]
    rep = rk.check_sum_identity(
        [yll, yld], daly, keys=keys, value_col="value", rel_tol=1e-9
    )
    assert rep.ok
    assert rep.n_compared == len(daly)
    bad = rk.check_sum_identity(
        [yll, yld], _measure(children, 4.0), keys=keys, value_col="value", rel_tol=1e-3
    )
    assert not bad.ok
    assert bad.max_rel_diff == pytest.approx(0.125)


def test_sum_identity_aligns_outer_with_zeros_and_excludes_at_a_coarser_grain() -> None:
    children = _children()
    yll = _measure(children, 3.0)
    yld = _measure(children, 0.5).iloc[:-1]  # one cell absent -> treated as 0
    daly = _measure(children, 3.5)
    keys = ["location_id", *KEYS]
    rep = rk.check_sum_identity(
        [yll, yld], daly, keys=keys, value_col="value", rel_tol=1e-9
    )
    assert not rep.ok
    missing_cell = _measure(children, 0.5).iloc[[-1]]
    excluded = missing_cell[["year_id", "age_group_id"]]
    rep2 = rk.check_sum_identity(
        [yll, yld],
        daly,
        keys=keys,
        value_col="value",
        rel_tol=1e-9,
        exclude=excluded,
        exclude_on=["year_id", "age_group_id"],
    )
    assert rep2.ok
    assert rep2.n_excluded == int(
        (
            (daly.year_id == missing_cell.year_id.iloc[0])
            & (daly.age_group_id == missing_cell.age_group_id.iloc[0])
        ).sum()
    )


def test_sum_identity_refusals() -> None:
    children = _children()
    keys = ["location_id", *KEYS]
    with pytest.raises(ValueError, match="at least one part"):
        rk.check_sum_identity(
            [], _measure(children, 1.0), keys=keys, value_col="value", rel_tol=1e-9
        )
    with pytest.raises(ValueError, match=r"parts\[0\] lacks"):
        rk.check_sum_identity(
            [_measure(children, 1.0).drop(columns="value")],
            _measure(children, 1.0),
            keys=keys,
            value_col="value",
            rel_tol=1e-9,
        )
    with pytest.raises(ValueError, match="total lacks"):
        rk.check_sum_identity(
            [_measure(children, 1.0)],
            _measure(children, 1.0).drop(columns="value"),
            keys=keys,
            value_col="value",
            rel_tol=1e-9,
        )
