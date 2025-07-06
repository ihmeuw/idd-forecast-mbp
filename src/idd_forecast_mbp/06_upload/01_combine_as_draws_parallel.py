import getpass
import uuid
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import geopandas as gpd  # type: ignore
from idd_forecast_mbp import constants as rfc

repo_name = rfc.repo_name
package_name = rfc.package_name


fhs_flag = 1
run_date = "2025_07_03"

# Script directory
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "06_upload"

draws = rfc.draws

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

wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_combine_draws_{wf_uuid}"
tool = Tool(name=tool_name)

# Create a workflow
workflow = tool.create_workflow(
    name=f"{tool_name}_workflow_{wf_uuid}",
    max_concurrently_running=10000,  # Adjust based on system capacity
)

# Compute resources
workflow.set_default_compute_resources_from_dict(
    cluster_name="slurm",
    dictionary={
        "memory": "15G",
        "cores": 1,
        "runtime": "60m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    }
)

if fhs_flag == 1:
    memory = "50G"
else:
    memory = "100G"

# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name="final_cause_aggregation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": memory,
        "cores": 5,
        "runtime": "60m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/combine_as_draws.py "
        "--cause {{cause}} "
        "--ssp_scenario {{ssp_scenario}} "
        "--dah_scenario {{dah_scenario}} "
        "--measure {{measure}} "
        "--metric {{metric}} "
        "--fhs_flag {{fhs_flag}} "
        "--run_date {{run_date}}"
    ).format(script_root=SCRIPT_ROOT),
    node_args=["cause", "ssp_scenario", "dah_scenario", "measure", "metric", "fhs_flag", "run_date"],
    task_args=[],
    op_args=[],
)

causes_to_process = rfc.cause_map
# causes_to_process = ['dengue']
# Add tasks
tasks = []
if fhs_flag == 1:
    for cause in causes_to_process:
        for ssp_scenario in rfc.ssp_scenarios:
            for measure in rfc.measure_map:
                if cause == "malaria":
                    # Create the primary task
                    task = task_template.create_task(
                        cause=cause,
                        ssp_scenario=ssp_scenario,
                        dah_scenario='Baseline',
                        measure=measure,
                        metric='rate',
                        fhs_flag=fhs_flag,
                        run_date=run_date
                    )
                    tasks.append(task)
                else:
                    dah_scenario = None
                    # Create the primary task
                    task = task_template.create_task(
                        cause=cause,
                        ssp_scenario=ssp_scenario,
                        dah_scenario=None,
                        measure=measure,
                        metric='rate',
                        fhs_flag=fhs_flag,
                        run_date=run_date
                    )
                    tasks.append(task)
else:
    for cause in rfc.cause_map:
        for ssp_scenario in rfc.ssp_scenarios:
            for measure in rfc.measure_map:
                for metric in rfc.metric_map:
                    if cause == "malaria":
                        for dah_scenario in rfc.dah_scenarios:
                            # Create the primary task
                            task = task_template.create_task(
                                cause=cause,
                                ssp_scenario=ssp_scenario,
                                dah_scenario=dah_scenario,
                                measure=measure,
                                metric=metric,
                                fhs_flag=fhs_flag,
                                run_date=run_date
                            )
                            tasks.append(task)
                    else:
                        dah_scenario = None
                        # Create the primary task
                        task = task_template.create_task(
                            cause=cause,
                            ssp_scenario=ssp_scenario,
                            dah_scenario=dah_scenario,
                            measure=measure,
                            metric=metric,
                            fhs_flag=fhs_flag,
                            run_date=run_date
                        )
                        tasks.append(task)

print(f"Number of tasks: {len(tasks)}")

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
