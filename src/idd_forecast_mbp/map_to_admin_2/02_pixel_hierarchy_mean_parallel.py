import getpass
import uuid
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import geopandas as gpd  # type: ignore
from rra_flooding import constants as rfc
from rra_flooding.helper_functions import load_yaml_dictionary

# Script directory
SCRIPT_ROOT = rfc.REPO_ROOT / "rra-flooding" / "src" / "rra_flooding" / "aggregate"
BASE_PATH = rfc.MODEL_ROOT / "output"
YAML_PATH = rfc.REPO_ROOT / "rra-flooding" / "src" / "rra_flooding" / "VARIABLE_DICT.yaml"


modeling_frame = gpd.read_parquet("/mnt/team/rapidresponse/pub/population-model/ihmepop_results/2025_03_22/modeling_frame.parquet")
block_keys = modeling_frame["block_key"].unique()
root = Path("/mnt/team/rapidresponse/pub/flooding/results/output/raw-results")

hierarchies = ["lsae_1209", "gbd_2021"]
# hierarchies = ["gbd_2021"]
models = ["ACCESS-CM2", "EC-Earth3", "INM-CM5-0", "MIROC6", "IPSL-CM6A-LR", "NorESM2-MM", "MRI-ESM2-0"]
scenarios = ["ssp126", "ssp245", "ssp585"]

VARIABLE_DICT = load_yaml_dictionary(YAML_PATH)


# Jobmon setup
user = getpass.getuser()

log_dir = Path(f"/mnt/share/homes/{user}/flood/")
log_dir.mkdir(parents=True, exist_ok=True)
# Create directories for stdout and stderr
stdout_dir = log_dir / "stdout"
stderr_dir = log_dir / "stderr"
stdout_dir.mkdir(parents=True, exist_ok=True)
stderr_dir.mkdir(parents=True, exist_ok=True)

# Project
project = "proj_lsae"  # Adjust this to your project name if needed


wf_uuid = uuid.uuid4()
tool = Tool(name="flood_model")

# Create a workflow
workflow = tool.create_workflow(
    name=f"pixel_workflow_{wf_uuid}",
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
        "--variable {{variable}} "
        "--adjustment_num {{adjustment_num}} "
        "--hierarchy {{hierarchy}} "
        "--scenario {{scenario}} "
        "--model {{model}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["hierarchy", "scenario", "model", "variable", "adjustment_num"],
    task_args=[],
    op_args=[],
)

# Define the second task template (aggregating results)
aggregate_task_template = tool.get_task_template(
    template_name="hierarchy_aggregation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "50G",
        "cores": 1,
        "runtime": "120m",
        "queue": "all.q",
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/pixel_mean_generation.py "
        "--variable {{variable}} "
        "--adjustment_num {{adjustment_num}} "
        "--hierarchy {{hierarchy}} "
        "--scenario {{scenario}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["hierarchy", "scenario", "variable", "adjustment_num"],
    task_args=[],
    op_args=[],
)

# Store tasks
tasks = []
aggregation_tasks = {}
# Add tasks
tasks = []
for variable in VARIABLE_DICT.keys():
    num_adjustments = len(VARIABLE_DICT[variable])
    for i in range(num_adjustments):
        for hierarchy in hierarchies:
            for model in models:
                for scenario in scenarios:
                    if model == "GFDL-CM4" and scenario == "ssp126":
                        continue
                    # Create the primary task
                    task = task_template.create_task(
                        variable = variable,
                        adjustment_num=i,
                        hierarchy=hierarchy,
                        scenario=scenario,
                        model=model,
                    )
                    tasks.append(task)

                    # Create an aggregation task only once per unique combination of hierarchy, scenario, and variant
                    agg_key = (variable, i, hierarchy, scenario)
                    if agg_key not in aggregation_tasks:
                        agg_task = aggregate_task_template.create_task(
                            variable = variable,
                            adjustment_num=i,
                            hierarchy=hierarchy,
                            scenario=scenario,
                        )
                        aggregation_tasks[agg_key] = agg_task  # Store it to reuse

# ✅ Add tasks to workflow
workflow.add_tasks(tasks + list(aggregation_tasks.values()))

# ✅ Set dependencies AFTER tasks are in the workflow
for hierarchy in hierarchies:
    for scenario in scenarios:
        agg_key = (variable, i, hierarchy, scenario)
        if agg_key in aggregation_tasks:
            agg_task = aggregation_tasks[agg_key]
            
            # Get all pixel tasks that match the same hierarchy, scenario, and variant
            pixel_tasks = workflow.get_tasks_by_node_args(
                "hierarchy_generation", variable = variable, adjustment_num=i, hierarchy=hierarchy, scenario=scenario
            )
            
            # Add dependency: aggregation should wait for all matching pixel tasks
            for pixel_task in pixel_tasks:
                agg_task.add_upstream(pixel_task)

print("✅ Tasks successfully added to workflow.")
print(f"  - First-level tasks: {len(tasks)}")
print(f"  - Aggregation tasks: {len(aggregation_tasks)}")
print(f"  - Total workflow tasks: {len(tasks) + len(aggregation_tasks)}")

# Bind and submit workflow
try:
    workflow.bind()
    print(f"✅ Workflow successfully bound. ID: {workflow.workflow_id}")
    print(f"View in Jobmon GUI: https://jobmon-gui.ihme.washington.edu/#/workflow/{workflow.workflow_id}")
except Exception as e:
    print(f"❌ Workflow binding failed: {e}")

try:
    status = workflow.run()
    print(f"✅ Workflow {workflow.workflow_id} completed with status {status}.")
except Exception as e:
    print(f"❌ Workflow execution failed: {e}")
