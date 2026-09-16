"""Register the data root's legacy snapshots per .claude/FREEZE_LIST.md (2026-09-16).

    .venv/bin/python scripts/register_freeze_list.py            # dry run: print the plan
    .venv/bin/python scripts/register_freeze_list.py --apply    # register, promote, label, copy models
    .venv/bin/python scripts/register_freeze_list.py --only forecast_outputs --apply

Nothing moves and nothing is deleted. For each node: existing dated directories are
registered as legacy snapshots (`register_existing`), the agreed one is promoted (which
also reconciles the `current` symlink), labels are attached. The three fitted models kept
from the flat 03-modeling_data/*.RData files are COPIED into the new models node as legacy
snapshots (RData + a run.json built from the old registry record). Already-registered
directories and existing labels are skipped, so the script can be re-run. DROP and LEAVE
items from the freeze list are not touched here.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import click
from idd_tools.versions import label, promote, read_registry, register_existing

from idd_forecast_mbp import constants as mbpc

if TYPE_CHECKING:
    from pathlib import Path

FIRST_SUB = "first-submission (2025) inputs in the pre-restructure layout"


@dataclass(frozen=True)
class Entry:
    directory: str
    description: str
    current: bool = False
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class NodePlan:
    node: str  # relative to MODEL_ROOT
    entries: tuple[Entry, ...]


def _node(rel: str, *entries: Entry) -> NodePlan:
    return NodePlan(rel, tuple(entries))


def _cur(directory: str, description: str, *labels: str) -> Entry:
    return Entry(directory, description, current=True, labels=tuple(labels))


def _keep(directory: str, description: str, *labels: str) -> Entry:
    return Entry(directory, description, current=False, labels=tuple(labels))


def _pre_restructure(rel: str) -> NodePlan:
    return _node(rel, _cur("pre_restructure", FIRST_SUB, "first_submission"))


PRODUCT_ARMS = (
    "2026_07_20_hybrid_fghjul",
    "2026_07_31_full_model_selection_results",
    "2026_07_31_full_model_selection_results__asstruct_hold2023",
    "2026_07_31_full_model_selection_results__denom_hold2023",
    "2026_07_31_full_model_selection_results__gdppc_hold2023",
    "2026_07_31_full_model_selection_results__gdpscen",
    "2026_07_31_full_model_selection_results__gdpscen__asstruct_hold2023",
    "2026_07_31_full_model_selection_results__gdpscen__denom_hold2023",
    "2026_07_31_full_model_selection_results__gdpscen__gdppc_hold2023",
)

PLAN: tuple[NodePlan, ...] = (
    _node(
        "01-raw_data/gbd", _cur("20260713", "GBD 2023 pull (constants.GBD_DATA_DATE)")
    ),
    _node(
        "01-raw_data/malaria_vaccine_coverage",
        _cur(
            "20260805",
            "vaccine coverage handoff 2026-08-05 (LME projection to 2100, not observed)",
        ),
    ),
    _node(
        "01-raw_data/malaria_vaccine_efficacy",
        _cur("20260824", "VE anchors received 2026-08-24"),
    ),
    _node(
        "02-processed_data/GBD2023/lsae_1285",
        _cur(
            "20260527", "stage-01 pixel aggregates on the 2026_05_15 gridded population"
        ),
    ),
    _node(
        "02-processed_data/covariates/dah",
        _cur("20260527", "malaria DAH from FGH 2026 July (corrected future DAH)"),
        _keep("pre_restructure", FIRST_SUB, "first_submission"),
    ),
    _node(
        "02-processed_data/covariates/gdppc/lsae_1285",
        _cur("20260803", "GDP per capita, V5 reference scenario for all RCPs"),
        _keep(
            "20260527",
            "GDP per capita behind forecast_inputs 20260527 (goalkeepers run)",
        ),
    ),
    _node(
        "02-processed_data/covariates/ldipc/lsae_1285",
        _cur("20260527", "LDI per capita, 2026-05-27 rebuild"),
    ),
    _node(
        "02-processed_data/covariates/med_consumppc/lsae_1285",
        _cur("20260527", "medical consumption per capita, 2026-05-27 rebuild"),
    ),
    _node(
        "02-processed_data/hierarchy/lsae_1285",
        _cur(
            "20260405",
            "full 2023 hierarchy at lsae_1285; unchanged by the gridded-population rebuild",
            "first_submission",
        ),
    ),
    _node(
        "02-processed_data/population/lsae_1285",
        _cur(
            "20260527",
            "all-age and age/sex population on the 2026_05_15 gridded population",
        ),
    ),
    _node(
        "02-processed_data/malaria/raked_aa/lsae_1285",
        _cur("20260527", "malaria all-age admin-2 raked to GBD, 2026-05-27 rebuild"),
    ),
    _node(
        "02-processed_data/malaria/raked_as/lsae_1285",
        _cur("20260527", "malaria age/sex admin-2 raked to GBD, 2026-05-27 rebuild"),
    ),
    _node(
        "02-processed_data/dengue/raked_aa/lsae_1285",
        _cur("20260527", "dengue all-age admin-2 raked to GBD, 2026-05-27 rebuild"),
    ),
    _node(
        "02-processed_data/dengue/raked_as/lsae_1285",
        _cur("20260527", "dengue age/sex admin-2 raked to GBD, 2026-05-27 rebuild"),
    ),
    _node(
        "02-processed_data/malaria_vaccine_efficacy",
        _cur("20260527", "VE curves built from VE_ANCHORS.yaml"),
    ),
    _node(
        "02-processed_data/urban/lsae_1285",
        _cur("20260527", "urban fraction covariate at lsae_1285"),
    ),
    _pre_restructure("02-processed_data/hierarchy/lsae_1209"),
    _pre_restructure("02-processed_data/population/lsae_1209"),
    _pre_restructure("02-processed_data/malaria/raked_aa/lsae_1209"),
    _pre_restructure("02-processed_data/malaria/raked_as/lsae_1209"),
    _pre_restructure("02-processed_data/dengue/raked_aa/lsae_1209"),
    _pre_restructure("02-processed_data/dengue/raked_as/lsae_1209"),
    _pre_restructure("03-modeling_data/covariates/covariate_means/lsae_1209"),
    _pre_restructure("03-modeling_data/dengue/modeling_dfs/lsae_1209"),
    _pre_restructure("03-modeling_data/malaria/modeling_dfs/lsae_1209"),
    _node(
        "03-modeling_data/malaria/past_inputs_nc/lsae_1285",
        _cur(
            "20260527",
            "malaria past inputs behind the selected model and the goalkeepers hybrids",
        ),
    ),
    _node(
        "03-modeling_data/malaria/scam_prelim/lsae_1285",
        _cur(
            "20260727_efs",
            "1,620-spec typed-space selection run; spec 1486 selected from it",
            "selected_2026_07_31",
        ),
        _keep(
            "20260720_efs", "1,620-spec selection run before the DAH-corrected rerun"
        ),
        _keep(
            "20260710_efs", "1,620-spec selection run after the 599278 census resize"
        ),
        _keep(
            "20260731_nbhd_rand10",
            "45-finalist random 10-fold CV (DEAD_ENDS 2026-09-16; not part of the selection)",
        ),
    ),
    _node(
        "03-modeling_data/dengue/fit_locations/lsae_1285",
        _cur("20260527", "dengue fit locations, 2026-05-27 rebuild"),
        _keep("20260708", "dengue fit locations, prototype era"),
    ),
    _node(
        "03-modeling_data/dengue/past_inputs_nc/lsae_1285",
        _cur("20260821_v2", "dengue past inputs at FHS grain (DECISIONS 2026-08-03)"),
        _keep("20260527", "dengue past inputs at admin-2 grain"),
        _keep("20260708", "dengue past inputs at admin-2 grain, prototype era"),
    ),
    _node(
        "03-modeling_data/dengue_cfr_model/lsae_1209",
        _cur(
            "20250703",
            "first-submission dengue CFR model",
            "first_submission",
            "pre_restructure",
        ),
    ),
    _node(
        "04-forecasting_data",
        _cur(
            "20260527",
            "non-draw-part forecast frames for both causes (07a), 2026-05-27 rebuild",
        ),
    ),
    _node(
        "04-forecasting_data/malaria/forecast_inputs/lsae_1285",
        _cur(
            "20260803",
            "malaria forecast inputs with V5 reference GDP; behind the 2026_07_31 gdpscen forecasts",
        ),
        _keep(
            "20260527",
            "malaria forecast inputs, DAH corrected in place 2026-07-20; behind the goalkeepers run",
        ),
    ),
    _node(
        "04-forecasting_data/malaria/forecast_outputs/lsae_1285",
        _cur(
            "2026_07_31_full_model_selection_results__gdpscen",
            "selected model, GDP-coupled forecast, 3 SSPs x Baseline/Constant DAH",
        ),
        _keep(
            "2026_07_31_full_model_selection_results",
            "selected model, GDP-decoupled forecast (sensitivity kept by DECISIONS 2026-08-03)",
        ),
        _keep(
            "2026_07_31_full_model_selection_results__gdppc_hold2023",
            "selected model, GDP held at 2023 (sensitivity)",
        ),
        _keep(
            "2026_07_31_full_model_selection_results__gdpscen__gdppc_hold2023",
            "selected model, GDP-coupled, GDP held at 2023 (sensitivity)",
        ),
        _keep(
            "2026_07_20_hybrid_fghjul",
            "goalkeepers 2026 delivered run: hybrid model on FGH July DAH",
            "goalkeepers_2026",
        ),
    ),
    _node(
        "04-forecasting_data/malaria/hybrid_deliverable/lsae_1285",
        _cur(
            "20260720",
            "goalkeepers 2026 incidence deliverable as delivered (from 2026_07_20_hybrid_fghjul)",
        ),
        _keep(
            "20260527",
            "original goalkeepers delivery; read as the old_delivered overlay",
            "goalkeepers_2026_original",
        ),
    ),
    _node(
        "04-forecasting_data/malaria/prediction_locations/lsae_1285",
        _cur("20260527", "malaria prediction locations, 2026-05-27 rebuild"),
    ),
    _node(
        "04-forecasting_data/malaria/vaccine_cohorts/lsae_1285",
        _cur("20260527", "vaccine cohort coverage x VE by location, year, age, sex"),
    ),
    _node(
        "04-forecasting_data/malaria/lsae_1209",
        _cur(
            "20250811",
            "first-submission malaria forecasts (behind upload 2025_08_28)",
            "first_submission",
        ),
    ),
    _node(
        "04-forecasting_data/dengue/lsae_1209",
        _cur(
            "20250811",
            "first-submission dengue forecasts (behind upload 2025_08_28)",
            "first_submission",
        ),
    ),
    _node(
        "04-forecasting_data/dengue/forecast_inputs/lsae_1285",
        _cur("20260527", "dengue forecast inputs, 2026-05-27 rebuild"),
        _keep("20260803", "dengue forecast inputs, August rebuild, never promoted"),
    ),
    _node(
        "04-forecasting_data/dengue/prediction_locations/lsae_1285",
        _cur("20260527", "dengue prediction locations, 2026-05-27 rebuild"),
    ),
    *[
        _node(
            f"05-products/malaria/lsae_1285/{arm}",
            _cur("20260527", f"finished products for forecast run {arm}"),
        )
        for arm in PRODUCT_ARMS
    ],
    _node(
        "05-products/dengue/lsae_1285",
        _keep("20260804", "dengue formulation-comparison products"),
    ),
    _node(
        "05-upload_data/upload_folders",
        _keep(
            "2025_08_28",
            "first-submission upload (constants.FIRST_SUBMISSION_UPLOAD_DATE)",
            "first_submission",
        ),
        _keep(
            "2025_08_11",
            "previous covariate upload carrying the cov_ds arms (constants.PREVIOUS_COVARIATE_UPLOAD_DATE)",
        ),
        _keep("GK_2025_11_02", "goalkeepers 2025 upload", "goalkeepers_2025"),
    ),
)

# Fitted models copied from the flat legacy files into the new node (name = legacy key).
MODELS_NODE_REL = str(mbpc.MAL_MODELS_NODE.relative_to(mbpc.MODEL_ROOT))
LEGACY_MODELS: tuple[Entry, ...] = (
    _cur(
        "2026_07_31_full_model_selection_results",
        "selected PfPR spec 1486 (full model selection, corrected DAH) + standard inc/mort",
        "full_model_selection_results",
    ),
    _keep(
        "2026_07_20_hybrid_fghjul",
        "goalkeepers 2026 hybrid model (2026_06_03 formulation, FGH July DAH)",
        "goalkeepers_2026",
    ),
    _keep(
        "2025_07_08",
        "first-submission model (pinned by the archived 2025 launcher, model_date 2025_07_08)",
        "first_submission",
    ),
)


def plan_rows(plan: tuple[NodePlan, ...] = PLAN) -> list[dict]:
    return [
        {
            "node": node.node,
            "directory": e.directory,
            "current": e.current,
            "labels": list(e.labels),
            "description": e.description,
        }
        for node in plan
        for e in node.entries
    ]


def check_plan(plan: tuple[NodePlan, ...] = PLAN) -> list[str]:
    """Internal consistency: one current per node, no duplicate directories or labels."""
    problems = []
    seen_nodes: set[str] = set()
    for node in plan:
        if node.node in seen_nodes:
            problems.append(f"node listed twice: {node.node}")
        seen_nodes.add(node.node)
        dirs = [e.directory for e in node.entries]
        if len(dirs) != len(set(dirs)):
            problems.append(f"duplicate directory in {node.node}")
        if sum(e.current for e in node.entries) > 1:
            problems.append(f"more than one current in {node.node}")
        labels = [lab for e in node.entries for lab in e.labels]
        if len(labels) != len(set(labels)):
            problems.append(f"duplicate label in {node.node}")
    return problems


def legacy_record(key: str) -> dict | None:
    for rec in mbpc.read_malaria_model_registry():
        if str(rec.get("run_date")) == key:
            return rec
    return None


def _apply_labels(
    node_path: Path, entry: Entry, have: tuple[str, ...], *, apply: bool, echo
) -> None:
    for name in entry.labels:
        if name in have:
            echo(f"  = label {name} already on {entry.directory}")
            continue
        echo(f"  label {entry.directory} {name}")
        if apply:
            label(node_path, entry.directory, name)


def apply_node(
    node_path: Path, entries: tuple[Entry, ...], *, apply: bool, echo=click.echo
) -> None:
    registered = (
        {r.version: r for r in read_registry(node_path)}
        if (node_path / "registry.json").is_file()
        else {}
    )
    for e in entries:
        target = node_path / e.directory
        if not target.is_dir():
            if not apply and node_path == mbpc.MAL_MODELS_NODE:
                echo(f"  (would register {e.directory} after the copy)")
            else:
                echo(f"  !! missing directory, skipped: {target}")
            continue
        if e.directory in registered:
            echo(f"  = already registered: {e.directory}")
            if e.current and not registered[e.directory].current:
                echo(f"  promote {e.directory}")
                if apply:
                    promote(node_path, e.directory)
        else:
            echo(
                f"  register {e.directory}{' --current' if e.current else ''}: {e.description}"
            )
            if apply:
                register_existing(
                    node_path, e.directory, e.description, current=e.current
                )
        have = registered[e.directory].labels if e.directory in registered else ()
        _apply_labels(node_path, e, tuple(have), apply=apply, echo=echo)


def copy_legacy_models(*, apply: bool, echo=click.echo) -> None:
    node = mbpc.MAL_MODELS_NODE
    echo(f"\n## {MODELS_NODE_REL} (copies of legacy fitted models)")
    if apply:
        node.mkdir(parents=True, exist_ok=True)
    for e in LEGACY_MODELS:
        src = mbpc._MODELING_STAGE / f"{e.directory}_malaria_models.RData"  # noqa: SLF001 - stage roots are underscore-named in constants
        dst_dir = node / e.directory
        if not src.is_file():
            echo(f"  !! legacy RData missing, skipped: {src}")
            continue
        if (dst_dir / mbpc.MAL_MODELS_RDATA).is_file():
            echo(f"  = copy present: {dst_dir.name}")
        else:
            echo(
                f"  copy {src.name} ({src.stat().st_size / 1e6:.0f} MB) -> {dst_dir}/{mbpc.MAL_MODELS_RDATA} + run.json"
            )
            if apply:
                dst_dir.mkdir(parents=True, exist_ok=True)
                tmp = dst_dir / (mbpc.MAL_MODELS_RDATA + ".tmp")
                shutil.copy2(src, tmp)
                tmp.replace(dst_dir / mbpc.MAL_MODELS_RDATA)
                rec = legacy_record(e.directory) or {
                    "run_date": e.directory,
                    "description": e.description,
                }
                run = {
                    **rec,
                    "legacy_source": str(src),
                    "copied_from_flat_registry": True,
                    "note": e.description,
                }
                (dst_dir / mbpc.MAL_MODELS_RUN_JSON).write_text(
                    json.dumps(run, indent=1, default=str)
                )
    apply_node(node, LEGACY_MODELS, apply=apply, echo=echo)


@click.command(help=__doc__)
@click.option(
    "--apply",
    "apply_",
    is_flag=True,
    help="Execute; default is a dry run that prints the plan.",
)
@click.option("--only", default=None, help="Substring filter on the node path.")
@click.option(
    "--skip-models", is_flag=True, help="Do not copy the legacy fitted models."
)
def main(*, apply_: bool, only: str | None, skip_models: bool) -> None:
    problems = check_plan()
    if problems:
        raise click.ClickException("plan is inconsistent:\n  " + "\n  ".join(problems))
    click.echo(
        f"{'APPLY' if apply_ else 'DRY RUN'}: {len(PLAN)} nodes, {len(plan_rows())} entries"
    )
    for node in PLAN:
        if only and only not in node.node:
            continue
        node_path = mbpc.MODEL_ROOT / node.node
        click.echo(f"\n## {node.node}")
        if not node_path.is_dir():
            click.echo(f"  !! node missing: {node_path}")
            continue
        apply_node(node_path, node.entries, apply=apply_)
    if not skip_models and (only is None or only in MODELS_NODE_REL):
        copy_legacy_models(apply=apply_)
    click.echo("\ndone" if apply_ else "\n(dry run; nothing changed)")


if __name__ == "__main__":
    main()
