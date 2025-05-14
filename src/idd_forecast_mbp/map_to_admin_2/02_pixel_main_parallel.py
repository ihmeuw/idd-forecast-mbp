import getpass
import uuid
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import geopandas as gpd # type: ignore
from idd_forecast_mbp import constants as rfc
from idd_forecast_mbp.helper_functions import load_yaml_dictionary, parse_yaml_dictionary

repo_name = rfc.repo_name
package_name = rfc.package_name

covariates= "test"

# Script directory
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "map_to_admin_2"
YAML_PATH = rfc.REPO_ROOT / repo_name / "src" / package_name / "COVARIATE_DICT.yaml"
COVARIATE_DICT = load_yaml_dictionary(YAML_PATH)

# Population block/tile stuff
modeling_frame = gpd.read_parquet("/mnt/team/rapidresponse/pub/population-model/ihmepop_results/2025_03_22/modeling_frame.parquet")
block_keys = modeling_frame["block_key"].unique()

# heirarchies = ["lsae_1209", "gbd_2021", "lsae_1285", "gbd_2023"]
heirarchies = ["lsae_1285", "gbd_2023"]
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
tool_name = f"{package_name}_pixel_generation"
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
    template_name="pixel_generation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "15G",
        "cores": 1,
        "runtime": "60m",
        "queue": "all.q",
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/pixel_main.py "
        "--covariate {{covariate}} "
        "--hiearchy {{hiearchy}} "
        "--block_key {{block_key}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=[ "hiearchy", "block_key", "covariate"],  #
    task_args=[], # Only variation is task-specific
    op_args=[],
)


# Add tasks
tasks = []
for covariate in COVARIATE_DICT.keys():
    covariate_dict = parse_yaml_dictionary(covariate)
    synoptic = covariate_dict['synoptic']
    if synoptic:
        for hiearchy in heirarchies:
            for block_key in block_keys:
                tasks.append(
                    task_template.create_task(
                        covariate=covariate,
                        hiearchy=hiearchy,
                        block_key=block_key
                    )
                )



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
