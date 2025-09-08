import getpass
import uuid
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import geopandas as gpd  # type: ignore
from idd_forecast_mbp import constants as rfc

repo_name = rfc.repo_name
package_name = rfc.package_name

# Script directory
SCRIPT_ROOT = rfc.REPO_ROOT / repo_name / "src" / package_name / "05_aggregation"

run_date = '2025_08_28'

ssp_scenarios = rfc.ssp_scenarios
dah_scenarios = rfc.dah_scenarios
draws = rfc.draws

hold_variables = {
    'malaria': ['DAH', 'flood', 'gdppc', 'suitability'],
    'dengue': ['gdppc', 'suitability', 'urban'],
}

run_hold_variables = True

dah_scenarios = rfc.dah_scenarios
dah_scenarios = ['Baseline','Constant']
# dah_scenarios = ['reference','better', 'worse']

causes = rfc.cause_map
causes = ['malaria', 'dengue']


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
tool_name = f"{package_name}_draw_level_cause_aggregation_{wf_uuid}"
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

# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name="as_cause_draw_aggregation",
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": "20G",
        "cores": 8,
        "runtime": "10m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/cause_as_aggregation_by_draw_raked.py "
        "--cause {{cause}} "
        "--ssp_scenario {{ssp_scenario}} "
        "--dah_scenario {{dah_scenario}} "
        "--measure {{measure}} "
        "--draw {{draw}} "
        "--hold_variable {{hold_variable}} "
        "--run_date {{run_date}} "
    ).format(script_root=SCRIPT_ROOT),
    node_args=["cause", "ssp_scenario", "dah_scenario", "measure", "draw", "hold_variable","run_date"],
    task_args=[],
    op_args=[],
)


# Add tasks
tasks = []
dah_scenarios = ['Baseline', 'Constant']
# for cause in cause_map:
for cause in causes:
    for ssp_scenario in ssp_scenarios:
        for measure in ['mortality', 'incidence']:
            for draw in draws:
                if cause == "malaria":
                    for dah_scenario in dah_scenarios:
                        # Create the primary task
                        task = task_template.create_task(
                            cause=cause,
                            ssp_scenario=ssp_scenario,
                            dah_scenario=dah_scenario,
                            measure=measure,
                            draw=draw,
                            run_date=run_date,
                            hold_variable='None'  # No hold variable for primary task
                        )
                        tasks.append(task)
                else:
                    task = task_template.create_task(
                        cause=cause,
                        ssp_scenario=ssp_scenario,
                        dah_scenario='None',
                        measure=measure,
                        draw=draw,
                        run_date=run_date,
                        hold_variable='None'  # No hold variable for primary task
                    )
                    tasks.append(task)
dah_scenarios = ['Baseline']
if run_hold_variables:
    for cause in causes:
        for hold_variable in hold_variables[cause]:
            for ssp_scenario in ssp_scenarios:
                for measure in ['mortality', 'incidence']:
                    for draw in draws:
                        if cause == "malaria":
                            for dah_scenario in dah_scenarios:
                                # Create the task with hold variable
                                task = task_template.create_task(
                                    cause=cause,
                                    ssp_scenario=ssp_scenario,
                                    dah_scenario=dah_scenario,
                                    measure=measure,
                                    draw=draw,
                                    run_date=run_date,
                                    hold_variable=hold_variable
                                )
                                tasks.append(task)
                        else:
                            # Create the task with hold variable
                            task = task_template.create_task(
                                cause=cause,
                                ssp_scenario=ssp_scenario,
                                dah_scenario='None',
                                measure=measure,
                                draw=draw,
                                run_date=run_date,
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
