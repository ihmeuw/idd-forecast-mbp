"""Output-node versioning for pipeline stages: the repo's adapter over ``idd_tools.versions``.

The contract (STANDARDS, Output management): a stage writes into ``<node>/working/``, or
``<node>/scratch/<label>/`` for a test run; nothing repoints ``current`` until a human
freezes, or the launcher was started with ``--current`` and the run succeeded. Readers
follow ``<node>/current/``.

What this module adds is one repo convention. Stage scripts that have no CLI receive
the launcher options through the environment::

    IDD_VERSIONS_CURRENT=1 IDD_VERSIONS_DESCRIPTION="stage 02, 2026 covariates" \\
        .venv/bin/python 02_data_prep/01_make_full_hierarchy.py

so :func:`finish_stage` works the same for a click script (pass its ``versions``) and a
plain script (read from the environment). Everything that mutates a node is
``idd_tools.versions``; nothing here creates a dated directory or touches a symlink.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import click
from idd_tools.versions import (
    FinishResult,
    VersionsOptions,
    finish_if_requested,
    preflight,
    resolve_target,
    scratch_dir,
    with_repo_dir,
    working_dir,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# The repo whose commit a snapshot records, regardless of the cwd the stage ran from.
REPO_DIR = Path(__file__).resolve().parents[3]

ENV_PREFIX = "IDD_VERSIONS_"
_ENV_FLAGS = {
    "FREEZE": "freeze",
    "CURRENT": "current",
    "TAG": "tag",
    "ALLOW_DIRTY": "allow_dirty",
    "CLEAN": "clean",
}
_ENV_VALUES = {"DESCRIPTION": "description", "LABEL": "label", "SCRATCH": "scratch"}
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


def write_path(node: Path | str, *, clean: bool = False) -> Path:
    """``<node>/working/``, created; the only directory a stage writes into."""
    return working_dir(node, clean=clean)


def scratch_path(node: Path | str, label: str) -> Path:
    """``<node>/scratch/<label>/``, created; a run that is a test and is never frozen."""
    return scratch_dir(node, label)


def read_path(node: Path | str) -> Path:
    """``<node>/current/``; refuses when no snapshot is current."""
    current = Path(node) / "current"
    if not current.exists():
        msg = (
            f"No current snapshot under {node}. Run the upstream stage, then freeze and "
            f"promote it (idd-versions {node} freeze '<why>' --current), or launch it with --current."
        )
        raise FileNotFoundError(msg)
    return current


def assert_artifact_ready(node: Path | str) -> Path:
    """The path a downstream stage should read: ``current``, else a non-empty ``working/`` with a warning.

    The fallback exists for a node that has never been frozen (a brand-new stage); it is
    not how chained rebuilds are meant to flow. Those launch upstream with ``--current``.
    """
    node = Path(node)
    current = node / "current"
    if current.exists():
        return current
    working = node / "working"
    if working.is_dir() and any(working.iterdir()):
        warnings.warn(
            f"No current snapshot under {node.name}; reading working/ (unfrozen output). "
            "Freeze and promote it, or launch the producing stage with --current.",
            stacklevel=2,
        )
        return working
    msg = (
        f"Artifact not ready: {node}\n"
        "Run the upstream stage that produces it; promote a snapshot or launch it with --current."
    )
    raise FileNotFoundError(msg)


def _flag(raw: str, name: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    msg = f"{ENV_PREFIX}{name} must be a boolean (1/0, true/false, yes/no, on/off); got {raw!r}"
    raise ValueError(msg)


def versions_from_env(environ: Mapping[str, str] | None = None) -> VersionsOptions:
    """The launcher options a plain (no-CLI) stage script was started with, from ``IDD_VERSIONS_*``.

    Unset means the defaults: write to ``working/``, freeze nothing. The same pre-flights
    a click launcher runs at parse time run here, so a missing description is refused
    before any compute.
    """
    env = os.environ if environ is None else environ
    values: dict[str, object] = {}
    for suffix, field in _ENV_FLAGS.items():
        raw = env.get(f"{ENV_PREFIX}{suffix}")
        if raw is not None:
            values[field] = _flag(raw, suffix)
    for suffix, field in _ENV_VALUES.items():
        raw = env.get(f"{ENV_PREFIX}{suffix}")
        if raw is not None and raw.strip():
            values[field] = raw.strip()
    versions = VersionsOptions(**values)  # type: ignore[arg-type]
    try:
        preflight(versions)
    except click.ClickException as exc:
        raise ValueError(exc.format_message()) from exc
    return versions


def stage_target(
    node: Path | str,
    versions: VersionsOptions | None = None,
    *,
    clean: bool | None = None,
) -> Path:
    """Where this run writes: ``resolve_target`` with the click options, or the environment's."""
    versions = versions if versions is not None else versions_from_env()
    return resolve_target(node, versions, clean=clean)


def finish_stage(
    node: Path | str, versions: VersionsOptions | None = None
) -> FinishResult | None:
    """The success path of a stage: freeze and promote ``working/`` if the launch asked for it.

    Pass the click ``versions`` when the script has one; a plain script leaves it ``None``
    and the ``IDD_VERSIONS_*`` variables decide. Provenance is recorded against this repo's
    checkout whatever the cwd. Returns ``None`` when nothing was requested.
    """
    versions = versions if versions is not None else versions_from_env()
    if versions.repo_dir is None:
        versions = with_repo_dir(versions, REPO_DIR)
    return finish_if_requested(node, versions)
