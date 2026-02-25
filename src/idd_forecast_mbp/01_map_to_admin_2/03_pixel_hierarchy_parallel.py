import getpass
import uuid
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import geopandas as gpd  # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.yaml_functions import load_yaml_dictionary

repo_name = rfc.repo_name
package_name = rfc.package_name

# Script directory
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "01_map_to_admin_2"
YAML_PATH = rfc.REPO_ROOT / repo_name / "src" / package_name / "COVARIATE_DICT.yaml"
COVARIATE_DICT = load_yaml_dictionary(YAML_PATH)

modeling_frame = gpd.read_parquet("/mnt/team/rapidresponse/pub/population-model/ihmepop_results/2025_03_22/modeling_frame.parquet")
block_keys = modeling_frame["block_key"].unique()
root = Path("/mnt/team/rapidresponse/pub/flooding/results/output/raw-results")

# heirarchies = ["lsae_1209", "gbd_2021", "lsae_1285", "gbd_2023"]
hierarchies = ["lsae_1209", "gbd_2023"]
scenarios = ["ssp126", "ssp245", "ssp585"]




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
project = "proj_rapidresponse"  # Adjust this to your project name if needed


wf_uuid = uuid.uuid4()
tool_name = f"{package_name}_hierarchy_generation"
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
    template_name="hierarchy_generation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "50G",
        "cores": 1,
        "runtime": "60m",
        "queue": "all.q",
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/pixel_hierarchy.py "
        "--covariate {{covariate}} "
        "--hierarchy {{hierarchy}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["covariate", "hierarchy"],
    task_args=[],
    op_args=[],
)


# Add tasks
tasks = []
for covariate in COVARIATE_DICT.keys():
    for hierarchy in hierarchies:
            
        # Create the primary task
        task = task_template.create_task(
            covariate=covariate,
            hierarchy=hierarchy,
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
