import getpass
import uuid
from jobmon.client.tool import Tool # type: ignore
from pathlib import Path
import geopandas as gpd # type: ignore
from idd_forecast_mbp import constants as mbpc
from idd_forecast_mbp.yaml_functions import load_yaml_dictionary, parse_yaml_dictionary

# Sibling module — importable because Python adds the script's dir to sys.path[0]
from block_utils import blocks_with_shapefile_intersections  # noqa: E402

repo_name = mbpc.repo_name
package_name = mbpc.package_name

thresholds = [300, 1500]

# Script directory
SCRIPT_ROOT = mbpc.REPO_ROOT / repo_name / "src" / package_name / "01_map_to_admin_2"

# Population block/tile stuff

modeling_frame_path = mbpc.MODELING_FRAME_PATH
modeling_frame = gpd.read_parquet(modeling_frame_path)
block_keys = modeling_frame["block_key"].unique()

hierarchies = mbpc.hierarchies
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
tool_name = f"{package_name}_urban_pixel_generation"
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
        "memory": "30G",
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
    template_name="pixel_urban_generation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "30G",
        "cores": 1,
        "runtime": "60m",
        "queue": "all.q",
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/pixel_urban_main.py "
        "--threshold {{threshold}} "
        "--hierarchy {{hierarchy}} "
        "--block_key {{block_key}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=[ "hierarchy", "block_key", "threshold"],  #
    task_args=[], # Only variation is task-specific
    op_args=[],
)


# Compute intersecting block_keys once per hierarchy — same for every threshold,
# so don't recompute inside the threshold loop. Lossless skip of blocks whose
# footprint doesn't overlap any admin polygon (open ocean, Antarctic interior).
intersecting_by_hier = {
    h: blocks_with_shapefile_intersections(h) for h in hierarchies
}

# Add tasks
tasks = []
for threshold in thresholds:
    for hierarchy in hierarchies:
        filtered_block_keys = [b for b in block_keys if b in intersecting_by_hier[hierarchy]]
        print(f"Creating tasks for threshold: {threshold}, hierarchy: {hierarchy} ({len(filtered_block_keys)}/{len(block_keys)} blocks)")
        for block_key in filtered_block_keys:
            tasks.append(
                task_template.create_task(
                    threshold=threshold,
                    hierarchy=hierarchy,
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
    status = workflow.run(seconds_until_timeout=60 * 60 * 24 * 3)  # 3 days
    print(f"Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow submission failed: {e}")
