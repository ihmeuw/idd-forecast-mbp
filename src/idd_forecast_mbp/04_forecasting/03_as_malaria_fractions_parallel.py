import getpass
import uuid
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import geopandas as gpd  # type: ignore
from idd_forecast_mbp import constants as rfc

repo_name = rfc.repo_name
package_name = rfc.package_name

# Script directory
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "04_forecasting"

malaria_hold_variables = ['DAH', 'flood', 'gdppc', 'suitability']
run_hold_variables = False

ssp_scenarios = rfc.ssp_scenarios
dah_scenarios = rfc.dah_scenarios
dah_scenarios = ["Baseline", "Constant"]
dah_scenarios = ['GK_reference_2025_11_02', 'GK_cut20_2025_11_02']
# dah_scenarios = ['reference', 'better', 'worse']
modeling_measure_map = rfc.modeling_measure_map
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
queue = 'all.q'

wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_malaria_level_5_age_sex_forecasting"
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
        "runtime": "5m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    }
)

# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name="malaria_as_calculation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "30G",
        "cores": 4,
        "runtime": "5m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/as_malaria_fractions.py "
        "--ssp_scenario {{ssp_scenario}} "
        "--dah_scenario {{dah_scenario}} "
        "--draw {{draw}} "
        "--hold_variable {{hold_variable}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["ssp_scenario", "dah_scenario", "draw", "hold_variable"],
    task_args=[],
    op_args=[],
)

# Add tasks
tasks = []
for ssp_scenario in ssp_scenarios:
    for draw in draws:
        for dah_scenario in dah_scenarios:
            task = task_template.create_task(
                ssp_scenario=ssp_scenario,
                dah_scenario=dah_scenario,
                draw=draw,
                hold_variable='None'
            )
            tasks.append(task)
            
if run_hold_variables:
    for hold_variable in malaria_hold_variables:
        for ssp_scenario in ssp_scenarios:
            for draw in draws:
                for dah_scenario in dah_scenarios:
                    task = task_template.create_task(
                        ssp_scenario=ssp_scenario,
                        dah_scenario=dah_scenario,
                        draw=draw,
                        hold_variable=hold_variable
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
