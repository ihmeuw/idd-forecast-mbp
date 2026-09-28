"""The measures the FHS return adds to our two, and where each comes from, as data.

Our forecast produces incidence and mortality. FHS returns, for the incidence and mortality
we submitted, its final incidence, deaths, YLLs and YLDs. The first-submission chain
(``05_aggregation/OLD_*``) raked our admin-2 incidence to their incidence and to their YLDs,
our admin-2 mortality to their deaths and to their YLLs, and summed YLL + YLD into DALYs.
This module is that table, so the driver reads it instead of branching on measure names,
and a test can refuse a source that is not one of ours (DECISIONS 2026-09-28).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

#: The two measures our forecast produces, in our names.
OUR_MEASURES: tuple[str, ...] = ("incidence", "mortality")


@dataclass(frozen=True)
class MeasureDerivation:
    """One FHS measure we produce by raking one of OUR measures to the FHS return for it."""

    #: The FHS measure name (also the FHS results subdirectory).
    output: str
    #: Which of our measures supplies the admin-2 draws that are raked to it.
    source: str


#: FHS measures obtained by raking, in the order the chain produced them.
FHS_RAKED: tuple[MeasureDerivation, ...] = (
    MeasureDerivation("incidence", "incidence"),
    MeasureDerivation("death", "mortality"),
    MeasureDerivation("yll", "mortality"),
    MeasureDerivation("yld", "incidence"),
)
#: FHS measures obtained by summing raked ones.
FHS_SUMMED: Mapping[str, tuple[str, ...]] = {"daly": ("yll", "yld")}


def validate_table(
    raked: Sequence[MeasureDerivation] = FHS_RAKED,
    summed: Mapping[str, tuple[str, ...]] = FHS_SUMMED,
    ours: Sequence[str] = OUR_MEASURES,
) -> None:
    """Refuse a table whose sources are not our measures or whose sums name unknown parts."""
    outputs = [d.output for d in raked]
    if len(set(outputs)) != len(outputs):
        msg = f"duplicate raked outputs: {outputs}"
        raise ValueError(msg)
    for d in raked:
        if d.source not in ours:
            msg = f"{d.output!r} is raked from {d.source!r}, which is not one of our measures {list(ours)}"
            raise ValueError(msg)
    for name, parts in summed.items():
        if name in outputs:
            msg = f"{name!r} is both raked and summed"
            raise ValueError(msg)
        unknown = [p for p in parts if p not in outputs]
        if unknown or not parts:
            msg = f"summed measure {name!r} names parts {list(parts)}; raked outputs are {outputs}"
            raise ValueError(msg)


validate_table()


def source_measure(output: str) -> str:
    """Our measure whose admin-2 draws are raked to the FHS ``output``."""
    for d in FHS_RAKED:
        if d.output == output:
            return d.source
    msg = f"{output!r} is not a raked FHS measure; raked: {[d.output for d in FHS_RAKED]}, summed: {sorted(FHS_SUMMED)}"
    raise ValueError(msg)


def raked_outputs_of(source: str) -> tuple[str, ...]:
    """The FHS measures obtained by raking our ``source`` measure (e.g. mortality -> death, yll)."""
    if source not in OUR_MEASURES:
        msg = f"{source!r} is not one of our measures {list(OUR_MEASURES)}"
        raise ValueError(msg)
    return tuple(d.output for d in FHS_RAKED if d.source == source)


def components_of(summed: str) -> tuple[str, ...]:
    """The raked measures that add up to ``summed`` (daly -> yll, yld)."""
    if summed not in FHS_SUMMED:
        msg = f"{summed!r} is not a summed FHS measure; summed: {sorted(FHS_SUMMED)}"
        raise ValueError(msg)
    return FHS_SUMMED[summed]


def all_fhs_measures() -> tuple[str, ...]:
    """Every FHS measure the chain produces, raked first, then summed."""
    return (*(d.output for d in FHS_RAKED), *FHS_SUMMED)


def sum_components(
    frames: Mapping[str, pd.DataFrame],
    summed: str,
    *,
    keys: Sequence[str],
    value_col: str,
) -> pd.DataFrame:
    """Add the component frames of ``summed`` cell by cell (outer alignment, absent cells 0).

    ``frames`` maps each component name to ``[*keys, value_col]``; every component must be
    present. Returns ``[*keys, value_col]``.
    """
    parts = components_of(summed)
    missing = [p for p in parts if p not in frames]
    if missing:
        msg = f"{summed!r} needs component frame(s) {missing}"
        raise ValueError(msg)
    key_list = list(keys)
    out: pd.DataFrame | None = None
    for part in parts:
        frame = frames[part]
        absent = [c for c in [*key_list, value_col] if c not in frame.columns]
        if absent:
            msg = f"component {part!r} lacks column(s) {absent}"
            raise ValueError(msg)
        g = frame.groupby(key_list, as_index=False)[value_col].sum()
        if out is None:
            out = g
            continue
        out = out.merge(g, on=key_list, how="outer", suffixes=("", "_r"))
        out[value_col] = out[value_col].fillna(0.0) + out[f"{value_col}_r"].fillna(0.0)
        out = out.drop(columns=f"{value_col}_r")
    if (
        out is None
    ):  # pragma: no cover - unreachable: validate_table refuses an empty component list
        msg = f"{summed!r} has no components"
        raise ValueError(msg)
    return out
