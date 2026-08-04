"""Jobmon launcher for the per-draw -> draw-dimensioned combine (idd-tools).

One task per (measure, ssp, hold), each running combine_as_draws_demo.py for a
single combo. Built so you can:
  * run the full 24-combo grid (2 measures x 3 ssps x 4 holds) at n=100,
  * run any SUBSET (filter --measures/--ssps/--holds, or jobmon --probe-only N),
  * give each task its own draw count (n) — runtime scales per-task from it,
  * run the timing PROBE: 3 tasks at n=10 / 25 / 50.

Run it from inside the idd-forecast-mbp env (sys.executable is reused as the
task interpreter, so no hardcoded python path):
    python combine_as_draws_jobmon.py --probe          # 3-task n-scaling probe
    python combine_as_draws_jobmon.py                   # full 24-combo grid, n=100
    python combine_as_draws_jobmon.py --measures incidence --ssps ssp126   # subset

Timing lands in the jobmon run record (result.run_record_path) + the GUI URL it
prints; per-task runtime is what answers "how long does each take / does it
scale with n".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from idd_tools.jobmon import (
    Task,
    TaskManifest,
    TaskTemplateSpec,
    submit_with_manifest,
)

HERE          = Path(__file__).resolve().parent
COMBINE       = HERE / "combine_as_draws_demo.py"
PY            = sys.executable                       # the idd-forecast-mbp env python
DENGUE_DIR    = "/mnt/team/idd/pub/forecast-mbp/04-forecasting_data/dengue/lsae_1209/20250811"
ALL_MEASURES  = ["incidence", "mortality"]
ALL_SSPS      = ["ssp126", "ssp245", "ssp585"]
ALL_HOLDS     = ["base", "_hold_gdppc", "_hold_suitability", "_hold_urban"]  # 'base' -> '' in combine

# One template: run the combine for a single combo. All five are per-task (node) args.
TEMPLATES = {
    "combine": TaskTemplateSpec(
        command_template=(
            f"{PY} {COMBINE} --measure {{measure}} --ssps {{ssp}} "
            f"--holds {{hold}} --n_draws {{n_draws}} --output_dir {{output_dir}}"
        ),
        node_args=["measure", "ssp", "hold", "n_draws", "output_dir"],
    )
}


def _task(idx: int, measure: str, ssp: str, hold: str, n: int, output_dir: str) -> Task:
    return Task(
        index=idx,
        task_id=f"{measure}_{ssp}_{hold}_n{n}",
        task_template="combine",
        task_args={"measure": measure, "ssp": ssp, "hold": hold,
                   "n_draws": str(n), "output_dir": output_dir},
    )


def build_manifest(args) -> tuple[TaskManifest, int]:
    if args.probe:
        # 3 tasks: SAME combo at n=10/25/50, each to its own dir (same filename
        # otherwise collides). Throwaway timing; written under a controlled project
        # subdir (NEVER /tmp — see CLAUDE.md § Output and I/O).
        combos = [("incidence", "ssp126", "_hold_urban", n) for n in (10, 25, 50)]
        tasks = [
            _task(i, m, s, h, n, f"{args.probe_dir}/n{n}")
            for i, (m, s, h, n) in enumerate(combos)
        ]
        return TaskManifest(workflow_name="combine_as_draws_probe", tasks=tasks), len(tasks)

    # grid (full or subset): cross product, all at the single --n-draws, into --output-dir
    combos = [(m, s, h) for m in args.measures for s in args.ssps for h in args.holds]
    tasks = [
        _task(i, m, s, h, args.n_draws, args.output_dir)
        for i, (m, s, h) in enumerate(combos)
    ]
    return TaskManifest(workflow_name="combine_as_draws", tasks=tasks), len(tasks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Jobmon: combine per-draw age-sex netCDFs into draw-dim files.")
    p.add_argument("--probe", action="store_true",
                   help="submit the 3-task timing probe (incidence/ssp126/_hold_urban at n=10,25,50).")
    p.add_argument("--measures", nargs="+", default=ALL_MEASURES, choices=ALL_MEASURES)
    p.add_argument("--ssps", nargs="+", default=ALL_SSPS, choices=ALL_SSPS)
    p.add_argument("--holds", nargs="+", default=ALL_HOLDS)
    p.add_argument("--n-draws", type=int, default=100, dest="n_draws",
                   help="draw count for grid-mode tasks (probe ignores this).")
    p.add_argument("--output-dir", default=DENGUE_DIR, dest="output_dir",
                   help="where combined .nc go (grid mode). Default: the dengue folder.")
    p.add_argument("--probe-dir", default=f"{DENGUE_DIR}/_combine_probe", dest="probe_dir",
                   help="throwaway output root for --probe tasks (controlled project subdir; never /tmp).")
    p.add_argument("--workflow-dir", default=f"{DENGUE_DIR}/_combine_jobmon", dest="workflow_dir",
                   help="jobmon bookkeeping (run record + logs).")
    p.add_argument("--concurrency", type=int, default=6,
                   help="max tasks running at once (NFS throttle).")
    p.add_argument("--memory", default="4G", help="per-task memory request.")
    p.add_argument("--runtime", default="50m",
                   help="per-task runtime limit (probe: n=100 ran ~32 min; scale_up_on_retry backstops).")
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--probe-only", type=int, default=None, dest="probe_only",
                   help="jobmon: scope to the first N tasks of whatever manifest is built.")
    args = p.parse_args()

    manifest, n_tasks = build_manifest(args)
    Path(args.workflow_dir).mkdir(parents=True, exist_ok=True)
    resources = {"memory": args.memory, "cores": args.cores, "runtime": args.runtime}
    print(f"Submitting '{manifest.workflow_name}': {n_tasks} task(s), "
          f"concurrency={args.concurrency}, resources={resources}"
          + (f", probe_only={args.probe_only}" if args.probe_only else ""))

    result = submit_with_manifest(
        manifest,
        output_dir=args.workflow_dir,
        templates=TEMPLATES,
        resources=resources,
        concurrency_limit=args.concurrency,
        probe_only=args.probe_only,
    )
    print(f"\nworkflow_id={result.workflow_id}  status={result.status}  "
          f"submitted={result.n_tasks_submitted}")
    print(f"run record: {result.run_record_path}")
