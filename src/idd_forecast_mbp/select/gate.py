"""The malaria selection gate as functions.

The gate is the human step between a ranked selection run and the model the forecast
loads: re-rank with the rank parameters changed, read the pick, record it, fit the picked
spec, flag the fitted model as current. Every action the notebook
``reports/model_selection/malaria_selection_gate.ipynb`` performs is one call here, so the
gate is tested without widgets and the notebook stays import-only (DECISIONS 2026-09-16,
SELECTION_PIPELINE_PLAN step 3).

Two buttons, in order:

* **Record pick** -- :meth:`GateSession.record_pick` writes ``selection_result.json`` with
  status ``recorded`` and the parameter values in force at the click, re-renders the report,
  then :func:`launch_fit` runs ``fit_selected_malaria_model.py`` into the models node's
  ``working/`` slot. Nothing is frozen by this button.
* **Flag best** -- :func:`flag_best` freezes that ``working/`` fit as a snapshot and promotes
  it, so ``current`` (the model the forecast loads) moves. It is enabled only once
  :func:`fit_status` finds a fit whose ``run.json`` points at this run's result file.

Nothing here decides a parameter: the committed config supplies every value and the gate
may change only the keys in :data:`OVERRIDABLE`.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from idd_tools.versions import freeze, promote, read_registry
from idd_tools.versions import label as attach_label

import idd_forecast_mbp
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.select.malaria_spec_design import build_universe
from idd_forecast_mbp.select.rank import (
    RESULT_FILE,
    RankParams,
    SelectionConfig,
    SelectionResult,
    load_config,
    load_summary,
    render_report,
    resolve_run_dir,
    run_selection,
    write_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from idd_tools.model_selection.space import ModelUniverse

PACKAGE_DIR = Path(idd_forecast_mbp.__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parents[1]
FIT_LAUNCHER = PACKAGE_DIR / "03_modeling" / "fit_selected_malaria_model.py"
QMD = REPO_DIR / "reports" / "model_selection" / "malaria_model_selection.qmd"
RECORDED_STATUS = "recorded"
# The rank-time judgments the gate's widgets may change; every other value stays as committed.
OVERRIDABLE: frozenset[str] = frozenset(
    {
        "tau_prune_threshold",
        "focus_metric",
        "tolerance_value",
        "profile_top_n",
        "cull_nonconverged",
    }
)
RESERVED_DIRS: frozenset[str] = frozenset({"working", "scratch"})


# --------------------------------------------------------------------------- the session
@dataclass
class GateSession:
    """One ranked run open in the gate: its config, run dir, summary and the typed universe."""

    config: SelectionConfig
    run_dir: Path
    summary: pd.DataFrame
    universe: ModelUniverse

    @classmethod
    def open(
        cls,
        config_path: str | Path,
        run_dir: str | Path | None = None,
        *,
        root: Path = mbpc._MODELING_STAGE,  # noqa: SLF001 - stage roots are underscore-named in constants
    ) -> GateSession:
        """Load the committed config and the run's ``selection_summary.parquet``."""
        cfg = load_config(config_path)
        target = resolve_run_dir(
            cfg.run_dir, None if run_dir is None else str(run_dir), root=root
        )
        return cls(
            config=cfg,
            run_dir=target,
            summary=load_summary(target),
            universe=build_universe(),
        )

    @property
    def result_path(self) -> Path:
        return self.run_dir / RESULT_FILE

    def params(self, **overrides: Any) -> RankParams:
        """The committed ``rank:`` parameters with the gate's overrides applied."""
        unknown = sorted(set(overrides) - OVERRIDABLE)
        if unknown:
            msg = (
                f"the gate may not change {unknown}; overridable: {sorted(OVERRIDABLE)}"
            )
            raise ValueError(msg)
        return dataclasses.replace(self.config.rank, **overrides)

    def rerank(self, **overrides: Any) -> SelectionResult:
        """Re-run the ranking with the overrides; cheap, so the widgets can call it live."""
        return run_selection(self.summary, self.universe, self.params(**overrides))

    def record_pick(
        self,
        result: SelectionResult,
        *,
        render: bool = True,
        quarto: str | Path | None = None,
        qmd: str | Path = QMD,
    ) -> dict[str, Path]:
        """Write the result with status ``recorded`` (its own parameters travel with it) and re-render the report."""
        paths = write_result(result, self.config, self.run_dir, status=RECORDED_STATUS)
        if render:
            paths["report"] = render_report(qmd, self.run_dir, quarto=quarto)
        return paths


# --------------------------------------------------------------------------- the final fit
def fit_command(  # noqa: PLR0913 - one argument per launcher knob
    config_path: str | Path,
    run_dir: str | Path,
    *,
    r_image: str,
    r_shell: str,
    python: str | Path = sys.executable,
    launcher: str | Path = FIT_LAUNCHER,
    past_inputs: str | Path | None = None,
    prep_script: str | Path | None = None,
) -> list[str]:
    """argv for ``fit_selected_malaria_model.py`` into the models node's ``working/`` slot.

    Deliberately no ``--freeze`` / ``--current``: the fit sits in ``working/`` until Flag best
    freezes and promotes it, so a fit that is never flagged is never a snapshot.
    """
    argv = [
        str(python),
        str(launcher),
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
        "--r-image",
        r_image,
        "--r-shell",
        r_shell,
    ]
    if past_inputs is not None:
        argv += ["--past-inputs", str(past_inputs)]
    if prep_script is not None:
        argv += ["--prep-script", str(prep_script)]
    return argv


def launch_fit(
    argv: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> subprocess.CompletedProcess[Any]:
    """Run the final fit and refuse to continue on a non-zero exit (nothing is frozen either way)."""
    proc = runner(list(argv), check=False)
    if proc.returncode != 0:
        msg = f"final fit failed with exit code {proc.returncode}; nothing frozen"
        raise RuntimeError(msg)
    return proc


# --------------------------------------------------------------------------- flag best
@dataclass(frozen=True)
class FitStatus:
    """Where the fit for a result file is: ``working``, a ``snapshot``, or ``none``."""

    location: str
    path: Path | None
    snapshot: str | None
    registered: bool
    result_file: str | None
    matches: bool


def _selection_result_file(fit_dir: Path) -> str | None:
    """The ``selection.result_file`` a fit's ``run.json`` points at, when both outputs exist."""
    run_json = fit_dir / mbpc.MAL_MODELS_RUN_JSON
    if not run_json.is_file() or not (fit_dir / mbpc.MAL_MODELS_RDATA).is_file():
        return None
    record = json.loads(run_json.read_text())
    selection = record.get("selection") or {}
    result_file = selection.get("result_file")
    if isinstance(
        result_file, list
    ):  # jsonlite writes a length-1 vector as a list unless unboxed
        result_file = result_file[0] if result_file else None
    return str(result_file) if result_file else None


def _no_fit() -> FitStatus:
    return FitStatus(
        location="none",
        path=None,
        snapshot=None,
        registered=False,
        result_file=None,
        matches=False,
    )


def fit_status(models_node: str | Path, result_path: str | Path) -> FitStatus:
    """Find the fit produced from ``result_path``: ``working/`` first, then the newest snapshot."""
    node = Path(models_node)
    want = Path(result_path).resolve()

    def matches(result_file: str | None) -> bool:
        return result_file is not None and Path(result_file).resolve() == want

    working = node / "working"
    result_file = _selection_result_file(working) if working.is_dir() else None
    if result_file is not None:
        return FitStatus(
            location="working",
            path=working,
            snapshot=None,
            registered=False,
            result_file=result_file,
            matches=matches(result_file),
        )

    if not node.is_dir():
        return _no_fit()
    registered = (
        {rec.version for rec in read_registry(node)}
        if (node / "registry.json").is_file()
        else set()
    )
    snapshots = sorted(
        (
            d
            for d in node.iterdir()
            if d.is_dir() and not d.is_symlink() and d.name not in RESERVED_DIRS
        ),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for snap in snapshots:
        result_file = _selection_result_file(snap)
        if matches(result_file):
            return FitStatus(
                location="snapshot",
                path=snap,
                snapshot=snap.name,
                registered=snap.name in registered,
                result_file=result_file,
                matches=True,
            )
    return _no_fit()


def flag_best(  # noqa: PLR0913 - the three callables are injection points for tests
    models_node: str | Path,
    result_path: str | Path,
    *,
    description: str,
    label: str | None = None,
    freezer: Callable[..., Any] = freeze,
    promoter: Callable[..., Any] = promote,
    labeller: Callable[..., Any] = attach_label,
) -> str:
    """Make the fit for ``result_path`` the model the forecast loads; returns the snapshot name.

    A fit still in ``working/`` is frozen with ``current=True`` (and ``label`` if given); a fit
    already frozen is promoted (and labelled). Refuses when no fit points at ``result_path``,
    so a stale fit for another run can never be flagged from this gate.
    """
    status = fit_status(models_node, result_path)
    if not status.matches:
        msg = (
            f"no fit for {result_path} under {models_node} (found: {status.location}); "
            "Record pick launches the fit first"
        )
        raise RuntimeError(msg)
    if status.location == "working":
        frozen = freezer(models_node, description, label=label, current=True)
        return str(frozen.snapshot)
    snapshot = str(status.snapshot)
    promoter(models_node, snapshot)
    if label:
        labeller(models_node, snapshot, label)
    return snapshot
