import getpass
import uuid
import pandas as pd # type: ignore
from jobmon.client.status_commands import workflow_tasks, task_status # type: ignore
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
from idd_forecast_mbp import constants as rfc

repo_name = rfc.repo_name
package_name = rfc.package_name

# Script directory
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "05_aggregation"
run_date = '2025_08_28'


CAUSES = ["malaria", "dengue"]
SCENARIOS = [0, 75, 76]
MEASURES = ["death", "incidence", "yll", "yld"]
DRAWS = [i for i in range(100)]

# Jobmon setup
user = getpass.getuser()

log_dir = Path("/mnt/team/idd/pub/")
log_dir.mkdir(parents=True, exist_ok=True)
# Create directories for stdout and stderr
stdout_dir = log_dir / "stdout"
stderr_dir = log_dir / "stderr"
stdout_dir.mkdir(parents=True, exist_ok=True)
stderr_dir.mkdir(parents=True, exist_ok=True)

# Project
project = "proj_rapidresponse"  # Adjust this to your project name if needed
queue = 'long.q'

# create jobmon jobs
user = getpass.getuser()
wf_uuid = uuid.uuid4()

# Create a tool
tool = Tool(name="malaria_dengue_raking")


# Create a workflow, and set the executor
workflow = tool.create_workflow(
    name=f"malaria_dengue_raking_{wf_uuid}",
)


# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name="malaria_dengue_raking_task",
    default_cluster_name="slurm",
    default_compute_resources={
        "queue": queue,
        "cores": int(2),
        "memory": "10G",
        "runtime": "10m",
        "project": project,  # Ensure the project is set correctly
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/create_raked_outcomes.py "
        "--cause {{cause}} "
        "--scenario {{scenario}} "
        "--measure {{measure}} "
        "--draw {{draw}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["cause", "scenario", "measure", "draw"],
    task_args=[],
    op_args=[],
)


def check_if_path_draw_exists(cause, scenario, measure, draw):
    out_dir = Path("/mnt/team/rapidresponse/pub/malaria-denv/deliverables/2025_08_26_admin_2_counts/output/")

    SCENARIO_MAP = {
        0: "ssp245",
        75: "ssp126",
        76: "ssp585",
    }
    scenario = SCENARIO_MAP[scenario]
    if measure == "death":
        measure = "mortality"

    if cause == "malaria":
        dirname = (
            f"as_cause_{cause}_measure_{measure}_metric_count_"
            f"ssp_scenario_{scenario}_dah_scenario_Baseline_raked"
        )
    else:
        dirname = (
            f"as_cause_{cause}_measure_{measure}_metric_count_"
            f"ssp_scenario_{scenario}_raked"
        )
    filename = f"draw_{int(draw)}.nc"

    output_name = out_dir / dirname / "2025_09_08" / filename

    if output_name.exists():
        return True
    return False


tasks = []
for cause in CAUSES:
    for scenario in SCENARIOS:
        for measure in MEASURES:
            for draw in DRAWS:
                task = task_template.create_task(
                    cause=cause,
                    scenario=scenario,
                    measure=measure,
                    draw=draw
                )
                tasks.append(task)

print(f"Number of tasks to run: {len(tasks)}")

if tasks:
    workflow.add_tasks(tasks)
    print("✅ Tasks successfully added to workflow.")
else:
    print("⚠️ No tasks added to workflow. Check task generation.")

try:
    workflow.bind()
    print("✅ Workflow successfully bound.")
    print(f"Running workflow with ID {workflow.workflow_id}.")
    print("For full information see the Jobmon GUI:")
    print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")

try:
    status = workflow.run()
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow submission failed: {e}")