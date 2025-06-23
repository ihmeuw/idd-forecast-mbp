import getpass
import uuid
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import pandas as pd  # type: ignore
import geopandas as gpd  # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import load_yaml_dictionary, read_parquet_with_integer_ids

repo_name = rfc.repo_name
package_name = rfc.package_name

# Script directory
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "aggregation" 
PROCESSED_DATA_PATH = rfc.MODEL_ROOT / "02-processed_data"

ssp_scenarios = ["ssp126", "ssp245", "ssp585"]
dah_scenarios = ["Baseline"]
draws = [f"{i:03d}" for i in range(100)]
cause_ids = [345, 357]  # Malaria and Dengue
cause_ids = [345]  # Malaria and Dengue

UPLOAD_DATA_PATH = "/mnt/team/idd/pub/forecast-mbp/05-upload_data"
GBD_DATA_PATH = f"{UPLOAD_DATA_PATH}/age_specific_gbd"
fhs_hierarchy_df_path = f"{GBD_DATA_PATH}/fhs_hierarchy.parquet"
fhs_hierarchy_df = pd.read_parquet(fhs_hierarchy_df_path)
hierarchy_df_path = f'{PROCESSED_DATA_PATH}/full_hierarchy_lsae_1209.parquet'
hierarchy_df = read_parquet_with_integer_ids(hierarchy_df_path)
hierarchy_df = hierarchy_df[hierarchy_df["level"] == 5]

# Jobmon setup
user = getpass.getuser()

log_dir = Path(f"/mnt/share/homes/{user}/{package_name}/")
log_dir.mkdir(parents=True, exist_ok=True)
# Create directories for stdout and stderr
stdout_dir = log_dir / "stdout"
stderr_dir = log_dir / "stderr"
stdout_dir.mkdir(parents=True, exist_ok=True)
stderr_dir.mkdir(parents=True, exist_ok=True)

# Project
project = "proj_lsae"  # Adjust this to your project name if needed

wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_draw_level_aggregation_{wf_uuid}"
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
        "queue": "all.q",
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    }
)

# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name="location_specific_aggregation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "10G",
        "cores": 1,
        "runtime": "5m",
        "queue": "all.q",
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/all_draw_location_results.py "
        "--location_id {{location_id}} "
        "--ssp_scenario {{ssp_scenario}} "
        "--dah_scenario {{dah_scenario}} "
        "--cause_id {{cause_id}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["location_id", "ssp_scenario", "dah_scenario", "cause_id"],
    task_args=[],
    op_args=[],
)

# Add tasks
tasks = []
for cause_id in cause_ids:
    for ssp_scenario in ssp_scenarios:
        for location_id in hierarchy_df["location_id"].unique():
            if cause_id == 345:  # Malaria
                for dah_scenario in dah_scenarios:
                    # Create the primary task
                    task = task_template.create_task(
                        location_id=location_id,
                        ssp_scenario=ssp_scenario,
                        dah_scenario=dah_scenario,
                        cause_id=cause_id,
                    )
                    tasks.append(task)
            elif cause_id == 357:  # Dengue
                # Create the primary task without dah_scenario
                task = task_template.create_task(
                    location_id=location_id,
                    ssp_scenario=ssp_scenario,
                    dah_scenario=None,  # Dengue does not use dah_scenario
                    cause_id=cause_id,
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