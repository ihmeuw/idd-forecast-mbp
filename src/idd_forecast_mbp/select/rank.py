"""Rank a finalized malaria model-selection run and pick the parsimonious spec.

This is the notebook ``reports/03_modeling/rank_malaria_models_idd_tools.ipynb`` (cells 1
to 20) as functions, so the pick is reproducible from a run directory plus a committed
parameter file instead of from cells with the knobs inside them. Nothing here decides a
parameter: every judgment comes from :class:`RankParams`, loaded from YAML by
:func:`load_config`, which refuses to run when a key is missing.

Pipeline (one call each, so the gate can re-run the cheap tail with new parameters):

    cfg      = load_config(path)
    summary  = load_summary(run_dir)
    universe = build_universe()
    result   = run_selection(summary, universe, cfg.rank)
    write_result(result, cfg, run_dir)          # selection_result.json + parquets
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml
from idd_tools.model_selection import (
    Sample,
    borda_rank,
    consensus_rank,
    kendall_tau_matrix,
    pairwise_dominance_summary,
    pareto_frontier,
    parsimonious_selection,
    prune_redundant_metrics,
    select_metrics,
    topsis_rank,
    winner_profile,
)

from idd_forecast_mbp.lib.io.parquet import write_parquet
from idd_forecast_mbp.select.malaria_metrics import register_malaria_metrics, windowed
from idd_forecast_mbp.select.malaria_spec_design import (
    AXIS_ORDER,
    formula_text,
    n_scams,
    n_smooths,
    n_terms,
)

if TYPE_CHECKING:
    from idd_tools.model_selection.space import ModelUniverse

AXES: tuple[str, ...] = tuple(AXIS_ORDER)
WINDOW_SEP = "__"

# The focus metric decides the parsimony band; its direction is a property of the score,
# not a choice, so it is fixed here rather than in the config.
FOCUS_DIRECTION: dict[str, str] = {"topsis_score": "higher", "borda_score": "lower"}
METHOD_RANK_COL: dict[str, str] = {
    "borda": "borda_rank",
    "topsis": "topsis_rank",
    "dominance": "dominance_rank",
}
TOLERANCE_RULES: frozenset[str] = frozenset({"std_fraction"})

_TOP_KEYS = frozenset({"cause", "run_dir", "fit", "rank"})
_RANK_KEYS = frozenset(
    {
        "metric_sample",
        "metric_space",
        "tau_prune_threshold",
        "consensus_methods",
        "focus_metric",
        "tolerance",
        "complexity_order",
        "profile_top_n",
        "cull_nonconverged",
    }
)
_MAX_SHOWN = 10  # spec indices listed in a bridge error before truncating
RESULT_FILE = "selection_result.json"
RANKING_FILE = "ranking.parquet"
CANDIDATES_FILE = "candidates.parquet"


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class RankParams:
    """The rank-time judgments. No field has a default on purpose."""

    metric_sample: str
    metric_space: str
    tau_prune_threshold: float
    consensus_methods: tuple[str, ...]
    focus_metric: str
    tolerance_rule: str
    tolerance_value: float
    complexity_order: tuple[str, ...]
    profile_top_n: int
    cull_nonconverged: bool

    def __post_init__(self) -> None:
        if self.focus_metric not in FOCUS_DIRECTION:
            msg = f"focus_metric must be one of {sorted(FOCUS_DIRECTION)}; got {self.focus_metric!r}"
            raise ValueError(msg)
        unknown = [m for m in self.consensus_methods if m not in METHOD_RANK_COL]
        if unknown or not self.consensus_methods:
            msg = f"consensus_methods must be a non-empty subset of {sorted(METHOD_RANK_COL)}; got {list(self.consensus_methods)}"
            raise ValueError(msg)
        focus_method = self.focus_metric.removesuffix("_score")
        if focus_method not in self.consensus_methods:
            msg = f"focus_metric {self.focus_metric!r} needs {focus_method!r} in consensus_methods"
            raise ValueError(msg)
        if self.tolerance_rule not in TOLERANCE_RULES:
            msg = f"tolerance.rule must be one of {sorted(TOLERANCE_RULES)}; got {self.tolerance_rule!r}"
            raise ValueError(msg)
        if self.tolerance_value < 0:
            msg = "tolerance.value must be >= 0"
            raise ValueError(msg)
        if self.profile_top_n < 1:
            msg = "profile_top_n must be >= 1"
            raise ValueError(msg)
        if not self.complexity_order:
            msg = "complexity_order must name at least one column"
            raise ValueError(msg)
        Sample(self.metric_sample)  # raises ValueError on an unknown sample name

    @property
    def focus_direction(self) -> str:
        return FOCUS_DIRECTION[self.focus_metric]

    @property
    def rank_cols(self) -> tuple[str, ...]:
        return tuple(METHOD_RANK_COL[m] for m in self.consensus_methods)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_sample": self.metric_sample,
            "metric_space": self.metric_space,
            "tau_prune_threshold": self.tau_prune_threshold,
            "consensus_methods": list(self.consensus_methods),
            "focus_metric": self.focus_metric,
            "tolerance": {"rule": self.tolerance_rule, "value": self.tolerance_value},
            "complexity_order": list(self.complexity_order),
            "profile_top_n": self.profile_top_n,
            "cull_nonconverged": self.cull_nonconverged,
        }


@dataclass(frozen=True)
class SelectionConfig:
    cause: str
    run_dir: str
    fit: dict[str, Any]
    rank: RankParams
    source: Path


def rank_params_from_dict(raw: dict[str, Any]) -> RankParams:
    """Build :class:`RankParams` from the ``rank:`` mapping; unknown or missing keys refuse."""
    keys = set(raw)
    if keys != _RANK_KEYS:
        missing, unknown = sorted(_RANK_KEYS - keys), sorted(keys - _RANK_KEYS)
        msg = f"rank: section mismatch; missing={missing} unknown={unknown}"
        raise ValueError(msg)
    tol = raw["tolerance"]
    if not isinstance(tol, dict):
        msg = "rank.tolerance must be a mapping"
        raise TypeError(msg)
    if set(tol) != {"rule", "value"}:
        msg = "rank.tolerance must have exactly the keys rule, value"
        raise ValueError(msg)
    return RankParams(
        metric_sample=str(raw["metric_sample"]),
        metric_space=str(raw["metric_space"]),
        tau_prune_threshold=float(raw["tau_prune_threshold"]),
        consensus_methods=tuple(str(m) for m in raw["consensus_methods"]),
        focus_metric=str(raw["focus_metric"]),
        tolerance_rule=str(tol["rule"]),
        tolerance_value=float(tol["value"]),
        complexity_order=tuple(str(c) for c in raw["complexity_order"]),
        profile_top_n=int(raw["profile_top_n"]),
        cull_nonconverged=bool(raw["cull_nonconverged"]),
    )


def load_config(path: str | Path) -> SelectionConfig:
    """Load the committed selection config. A missing file or an unexpected shape refuses."""
    path = Path(path)
    if not path.is_file():
        msg = f"selection config not found: {path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        msg = f"{path}: top level must be a mapping; got {type(raw).__name__}"
        raise TypeError(msg)
    if set(raw) != _TOP_KEYS:
        msg = f"{path}: top-level keys must be exactly {sorted(_TOP_KEYS)}; got {sorted(raw)}"
        raise ValueError(msg)
    if not isinstance(raw["fit"], dict) or not isinstance(raw["rank"], dict):
        msg = f"{path}: fit: and rank: must be mappings"
        raise TypeError(msg)
    return SelectionConfig(
        cause=str(raw["cause"]),
        run_dir=str(raw["run_dir"]),
        fit=dict(raw["fit"]),
        rank=rank_params_from_dict(raw["rank"]),
        source=path,
    )


# --------------------------------------------------------------------------- inputs
def load_summary(run_dir: str | Path) -> pd.DataFrame:
    """Read ``selection_summary.parquet`` (finalize's output) from a run directory."""
    path = Path(run_dir) / "selection_summary.parquet"
    if not path.is_file():
        msg = f"no selection_summary.parquet in {run_dir}; run finalize_selection_run.py first"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def attach_universe(summary: pd.DataFrame, universe: ModelUniverse) -> pd.DataFrame:
    """Bridge ``spec_index`` to the typed model space (``configs[i] <-> spec_index i+1``).

    Adds the axis columns and the complexity counts the parsimony step orders on. Counts
    are pure functions of the config, so they are derived here rather than trusted from
    the summary (finalize does not carry ``n_scams``/``n_terms``).
    """
    if "spec_index" not in summary.columns:
        msg = "summary has no spec_index column"
        raise ValueError(msg)
    configs = pd.DataFrame(universe.configs)
    configs["spec_index"] = range(1, len(configs) + 1)
    df = summary.merge(configs, on="spec_index", how="left", validate="1:1")
    if not df[list(AXES)].notna().all().all():
        bad = df.loc[df[list(AXES)].isna().any(axis=1), "spec_index"].tolist()
        shown = bad[:_MAX_SHOWN]
        tail = "..." if len(bad) > _MAX_SHOWN else ""
        msg = f"spec_index -> Config bridge is incomplete for {shown}{tail}"
        raise ValueError(msg)
    cfg_of = {i + 1: c for i, c in enumerate(universe.configs)}
    df["n_scams"] = [n_scams(cfg_of[i]) for i in df["spec_index"]]
    df["n_terms"] = [n_terms(cfg_of[i]) for i in df["spec_index"]]
    if "n_smooths" not in df.columns:
        df["n_smooths"] = [n_smooths(cfg_of[i]) for i in df["spec_index"]]
    if "formula_text" not in df.columns:
        df["formula_text"] = [formula_text(cfg_of[i]) for i in df["spec_index"]]
    return df


def detect_windows(df: pd.DataFrame, metric_names: list[str]) -> list[str]:
    """The CV windows present as ``<window>__<metric>`` columns; must agree across metrics."""
    per_metric = [
        sorted(
            {
                c.split(WINDOW_SEP)[0]
                for c in df.columns
                if c.endswith(f"{WINDOW_SEP}{m}")
            }
        )
        for m in metric_names
    ]
    if not per_metric or not per_metric[0]:
        msg = f"no windowed columns found for metrics {metric_names}"
        raise ValueError(msg)
    if any(w != per_metric[0] for w in per_metric[1:]):
        msg = f"metrics disagree on windows: {dict(zip(metric_names, per_metric, strict=True))}"
        raise ValueError(msg)
    return per_metric[0]


# --------------------------------------------------------------------------- ranking
@dataclass
class SelectionResult:
    params: RankParams
    windows: list[str]
    criteria: dict[str, str]
    pruned_criteria: dict[str, str]
    prune_log: pd.DataFrame
    borda_cutpoint: int
    topsis_influence: pd.DataFrame
    consensus: pd.DataFrame
    pareto_spec_indices: list[int]
    profile: pd.DataFrame
    scored: pd.DataFrame
    tolerance: float
    anchor: pd.Series
    pick: pd.Series
    candidates: pd.DataFrame
    n_specs: int
    n_culled: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def run_selection(
    summary: pd.DataFrame, universe: ModelUniverse, params: RankParams
) -> SelectionResult:
    """Rank every spec on the windowed OOS criteria and return the parsimonious pick."""
    register_malaria_metrics()
    df = attach_universe(summary, universe)

    n_culled = 0
    if params.cull_nonconverged and "is_converged" in df.columns:
        before = len(df)
        df = df[df["is_converged"].fillna(value=False).astype(bool)].copy()
        n_culled = before - len(df)

    specs = select_metrics(
        sample=Sample(params.metric_sample), space=params.metric_space
    )
    if not specs:
        msg = f"no registered metrics with sample={params.metric_sample} space={params.metric_space}"
        raise ValueError(msg)
    metric_names = [s.name for s in specs]
    windows = detect_windows(df, metric_names)
    criteria = windowed(specs, windows)

    tau = kendall_tau_matrix(df, criteria)
    pruned, prune_log = prune_redundant_metrics(
        tau, criteria, threshold=params.tau_prune_threshold
    )

    borda_df, borda_cutpoint = borda_rank(df, pruned)
    topsis_df, topsis_influence = topsis_rank(df, pruned)
    pareto = pareto_frontier(df, pruned)
    dominance_df = pairwise_dominance_summary(df, pruned)

    ranks = (
        borda_df[["spec_index", "borda_rank", "borda_score", *pruned.keys()]]
        .merge(
            topsis_df[["spec_index", "topsis_rank", "topsis_score"]], on="spec_index"
        )
        .merge(dominance_df[["spec_index", "dominance_rank"]], on="spec_index")
    )
    id_cols = ["spec_index", "n_smooths", "n_scams", "n_terms", "formula_text", *AXES]
    focus_rank_col = METHOD_RANK_COL[params.focus_metric.removesuffix("_score")]
    consensus = (
        consensus_rank(ranks, rank_cols=params.rank_cols)
        .merge(df[id_cols], on="spec_index")
        .sort_values(focus_rank_col, kind="stable")
        .reset_index(drop=True)
    )
    pareto_ids = [int(i) for i in pareto["spec_index"]]
    consensus["on_pareto"] = consensus["spec_index"].isin(set(pareto_ids))
    profile = winner_profile(consensus, id_cols=list(AXES), top_n=params.profile_top_n)

    scored = df.merge(topsis_df[["spec_index", "topsis_score"]], on="spec_index").merge(
        borda_df[["spec_index", "borda_score"]], on="spec_index"
    )
    tolerance = compute_tolerance(scored, params)
    picked = parsimonious_selection(
        scored,
        metric=params.focus_metric,
        complexity=list(params.complexity_order),
        tol=tolerance,
        direction=params.focus_direction,
        config_cols=list(AXES),
        space=universe.space,
    )
    return SelectionResult(
        params=params,
        windows=windows,
        criteria=dict(criteria),
        pruned_criteria={k: str(v) for k, v in pruned.items()},
        prune_log=prune_log,
        borda_cutpoint=int(borda_cutpoint),
        topsis_influence=topsis_influence,
        consensus=consensus,
        pareto_spec_indices=pareto_ids,
        profile=profile,
        scored=scored,
        tolerance=tolerance,
        anchor=picked["anchor"],
        pick=picked["pick"],
        candidates=picked["candidates"],
        n_specs=len(df),
        n_culled=n_culled,
    )


def compute_tolerance(scored: pd.DataFrame, params: RankParams) -> float:
    """The parsimony band under the configured rule (only ``std_fraction`` exists so far)."""
    if params.tolerance_rule == "std_fraction":
        return float(scored[params.focus_metric].std() * params.tolerance_value)
    msg = f"unknown tolerance rule {params.tolerance_rule!r}"
    raise ValueError(msg)


# --------------------------------------------------------------------------- outputs
def file_fingerprint(path: str | Path) -> str:
    """``sha256:<hex>`` of a file's bytes; ties a result to the exact spec table it ranked."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def git_info(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Short HEAD and dirtiness of the code that produced a result; never raises."""
    cwd = str(repo_dir) if repo_dir else str(Path(__file__).resolve().parent)
    git = shutil.which("git")
    if git is None:
        return {"commit": "unknown", "dirty": None}
    try:
        head = subprocess.check_output(
            [git, "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [git, "status", "--porcelain"],
                cwd=cwd,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": head, "dirty": dirty}


def _row_record(row: pd.Series, extra_cols: list[str]) -> dict[str, Any]:
    keep = [
        "spec_index",
        "n_smooths",
        "n_scams",
        "n_terms",
        "formula_text",
        *AXES,
        *extra_cols,
    ]
    out: dict[str, Any] = {}
    for c in keep:
        if c not in row.index:
            continue
        v = row[c]
        out[c] = (
            int(v)
            if c in ("spec_index", "n_smooths", "n_scams", "n_terms")
            else (
                float(v)
                if hasattr(v, "__float__") and not isinstance(v, str)
                else str(v)
            )
        )
    return out


def result_record(
    result: SelectionResult,
    config: SelectionConfig,
    run_dir: str | Path,
    *,
    status: str = "tentative",
) -> dict[str, Any]:
    """The JSON-serializable record of one selection: inputs, parameters, and the pick."""
    run_dir = Path(run_dir)
    spec_table = run_dir / "spec_table.parquet"
    scores = ["topsis_score", "borda_score"]
    top = result.consensus.head(result.params.profile_top_n)
    top_cols = [
        "spec_index",
        "n_smooths",
        "n_scams",
        "n_terms",
        "consensus_rank",
        "on_pareto",
        "borda_rank",
        "borda_score",
        "topsis_rank",
        "topsis_score",
        "dominance_rank",
        "formula_text",
    ]
    return {
        "status": status,
        "cause": config.cause,
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_dir_config": config.run_dir,
        "spec_table_fingerprint": file_fingerprint(spec_table)
        if spec_table.is_file()
        else None,
        "n_specs": result.n_specs,
        "n_culled": result.n_culled,
        "code": git_info(),
        "config_source": str(config.source),
        "fit": config.fit,
        "rank": result.params.to_dict(),
        "windows": result.windows,
        "criteria": result.criteria,
        "pruned_criteria": result.pruned_criteria,
        "borda_cutpoint": result.borda_cutpoint,
        "tolerance": result.tolerance,
        "n_candidates": len(result.candidates),
        "anchor": _row_record(result.anchor, scores),
        "pick": _row_record(result.pick, scores),
        "candidates": [int(i) for i in result.candidates["spec_index"]],
        "pareto_spec_indices": result.pareto_spec_indices,
        "winner_profile": {
            str(k): {
                kk: (
                    int(vv)
                    if kk in ("mode_count", "n_unique")
                    else (float(vv) if kk == "mode_pct" else str(vv))
                )
                for kk, vv in v.items()
            }
            for k, v in result.profile.to_dict(orient="index").items()
        },
        "top": json.loads(
            top[[c for c in top_cols if c in top.columns]].to_json(orient="records")
        ),
    }


def _typed_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit dtypes for the parquet outputs (repo I/O rule: never rely on inference)."""
    out = frame.copy()
    for c in ("spec_index", "n_smooths", "n_scams", "n_terms"):
        if c in out.columns:
            out[c] = out[c].astype("int32")
    for c in ("formula_text", *AXES):
        if c in out.columns:
            out[c] = out[c].astype("string")
    for c in out.columns:
        if c.endswith(("_rank", "_score")) or c == "consensus_score":
            out[c] = out[c].astype("float64")
    if "on_pareto" in out.columns:
        out["on_pareto"] = out["on_pareto"].astype("bool")
    return out


def write_result(
    result: SelectionResult,
    config: SelectionConfig,
    run_dir: str | Path,
    *,
    status: str = "tentative",
) -> dict[str, Path]:
    """Write ``selection_result.json``, ``ranking.parquet`` and ``candidates.parquet`` into the run dir."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        msg = f"run dir does not exist: {run_dir}"
        raise FileNotFoundError(msg)
    paths = {
        "result": run_dir / RESULT_FILE,
        "ranking": run_dir / RANKING_FILE,
        "candidates": run_dir / CANDIDATES_FILE,
    }
    record = result_record(result, config, run_dir, status=status)
    tmp = paths["result"].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(paths["result"])
    write_parquet(_typed_ranking(result.consensus), paths["ranking"])
    cand_cols = [
        c
        for c in [
            "spec_index",
            "n_smooths",
            "n_scams",
            "n_terms",
            "topsis_score",
            "borda_score",
            "formula_text",
            *AXES,
        ]
        if c in result.candidates.columns
    ]
    write_parquet(_typed_ranking(result.candidates[cand_cols]), paths["candidates"])
    return paths


def read_result(run_dir: str | Path) -> dict[str, Any]:
    """Read back a written ``selection_result.json``."""
    path = Path(run_dir) / RESULT_FILE
    if not path.is_file():
        msg = f"no {RESULT_FILE} in {run_dir}"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def format_pick(result: SelectionResult) -> str:
    """The console summary printed by the CLI and the gate."""
    p = result.params
    a, k = result.anchor, result.pick
    lines = [
        f"{result.n_specs} specs; {len(result.windows)} windows x {len(result.criteria) // max(len(result.windows), 1)} metrics = {len(result.criteria)} criteria, {len(result.pruned_criteria)} after tau prune (>= {p.tau_prune_threshold})",
        f"tolerance +/-{result.tolerance:.4f} on {p.focus_metric} ({p.tolerance_rule} {p.tolerance_value}); {len(result.candidates)} candidates in the down-set",
        f"anchor : spec {int(a['spec_index'])}  n_smooths={int(a['n_smooths'])} n_scams={int(a['n_scams'])}  {p.focus_metric}={float(a[p.focus_metric]):.4f}",
        f"         {a['formula_text']}",
        f"pick   : spec {int(k['spec_index'])}  n_smooths={int(k['n_smooths'])} n_scams={int(k['n_scams'])}  {p.focus_metric}={float(k[p.focus_metric]):.4f}",
        f"         {k['formula_text']}",
    ]
    return "\n".join(lines)
