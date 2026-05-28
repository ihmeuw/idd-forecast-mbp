import getpass
import uuid

from idd_forecast_mbp import constants as mbpc
from jobmon.client.tool import Tool  # type: ignore

repo_name = mbpc.repo_name
package_name = mbpc.package_name

SCRIPT_ROOT = mbpc.REPO_ROOT / repo_name / "src" / package_name / "06_upload"

run_date = "2025_08_28"
release_id = 9

causes = list(mbpc.cause_map.keys())
ssp_scenarios = mbpc.ssp_scenarios
dah_scenarios = list(mbpc.dah_scenarios.keys())

user = getpass.getuser()
log_dir = mbpc.MODEL_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
stdout_dir = log_dir / "stdout"
stderr_dir = log_dir / "stderr"
stdout_dir.mkdir(parents=True, exist_ok=True)
stderr_dir.mkdir(parents=True, exist_ok=True)

project = "proj_rapidresponse"
queue = "all.q"

wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_fhs_upload_{wf_uuid}"
tool = Tool(name=tool_name)

workflow = tool.create_workflow(
    name=f"{tool_name}_workflow_{wf_uuid}",
    max_concurrently_running=10000,
)

workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "50G",
        "cores": 4,
        "runtime": "120m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
)

task_template = tool.get_task_template(
    template_name=f"{repo_name}_06_07_fhs_upload",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "50G",
        "cores": 4,
        "runtime": "120m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/fhs_upload_as_draws.py "
        "--cause {{cause}} "
        "--ssp_scenario {{ssp_scenario}} "
        "--dah_scenario {{dah_scenario}} "
        "--measure {{measure}} "
        "--run_date {{run_date}} "
        "--release_id {{release_id}}"
    ).format(script_root=SCRIPT_ROOT),
    node_args=["cause", "ssp_scenario", "dah_scenario", "measure", "run_date", "release_id"],
    task_args=[],
    op_args=[],
)

tasks = []
for cause in causes:
    for ssp_scenario in ssp_scenarios:
        for measure in ["mortality", "incidence"]:
            if cause == "malaria":
                for dah_scenario in dah_scenarios:
                    task = task_template.create_task(
                        cause=cause,
                        ssp_scenario=ssp_scenario,
                        dah_scenario=dah_scenario,
                        measure=measure,
                        run_date=run_date,
                        release_id=release_id,
                    )
                    tasks.append(task)
            else:
                task = task_template.create_task(
                    cause=cause,
                    ssp_scenario=ssp_scenario,
                    dah_scenario="None",
                    measure=measure,
                    run_date=run_date,
                    release_id=release_id,
                )
                tasks.append(task)

print(f"Number of tasks: {len(tasks)}")

if tasks:
    workflow.add_tasks(tasks)
    print("Tasks successfully added to workflow.")
else:
    print("No tasks added to workflow. Check task generation.")

try:
    workflow.bind()
    print(f"Workflow bound. ID: {workflow.workflow_id}")
    print(f"https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
except Exception as e:
    print(f"Workflow binding failed: {e}")

try:
    status = workflow.run()
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"Workflow submission failed: {e}")
