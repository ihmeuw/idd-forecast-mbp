import getpass
import uuid
from jobmon.client.tool import Tool  # type: ignore
from pathlib import Path
import geopandas as gpd  # type: ignore
from idd_forecast_mbp import constants as rfc

repo_name = rfc.repo_name
package_name = rfc.package_name

hold_variables = {
    'malaria': ['DAH', 'flood', 'gdppc', 'suitability', 'population', 'as_structure'],
    'dengue': ['gdppc', 'suitability', 'urban', 'population', 'as_structure'],
}

hold_variables = {
    'malaria': ['population', 'as_structure'],
    'dengue': ['population', 'as_structure'],
}


run_hold_variables = True
run_base_variables = False



template_name = f'{repo_name}_06_03_create_and_combine'

run_date = "2025_07_24"
run_date = '2025_08_04'
run_date = '2025_08_28'
dah_scenarios = rfc.dah_scenarios
dah_scenarios = ['Baseline', 'Constant']
dah_scenarios = ['Baseline']
# dah_scenarios = ['reference', 'better', 'worse']
# measures = ['mortality', 'incidence', 'yll', 'yld', 'daly']
full_measure_map = rfc.full_measure_map
measures = full_measure_map

causes = rfc.cause_map
# causes = ['dengue']
ssp_scenarios = rfc.ssp_scenarios
# ssp_scenarios = ['ssp245']


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
tool_name = f"{package_name}_create_summaries_{wf_uuid}"
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

memory = "35G"

# Define the task template for processing each year batch
task_template = tool.get_task_template(
    template_name=template_name,
    default_cluster_name="slurm",
    default_compute_resources={
        "memory": memory,
        "cores": 10,
        "runtime": "75m",
        "queue": queue,
        "project": project,
        "stdout": str(stdout_dir),
        "stderr": str(stderr_dir),
    },
    command_template=(
        "python {script_root}/create_and_combine_aa_draws.py "
        "--cause {{cause}} "
        "--ssp_scenario {{ssp_scenario}} "
        "--dah_scenario {{dah_scenario}} "
        "--measure {{measure}} "
        "--hold_variable {{hold_variable}} "
        "--run_date {{run_date}}"
    ).format(script_root=SCRIPT_ROOT),
    node_args=["cause", "ssp_scenario", "dah_scenario", "measure", "hold_variable", "run_date"],
    task_args=[],
    op_args=[],
)


tasks = []
dah_scenarios = ['Baseline', 'Constant']
if run_base_variables:
    for cause in causes:
        for ssp_scenario in ssp_scenarios:
            for measure in measures:
                if cause == "malaria":
                    for dah_scenario in dah_scenarios:
                        # Create the primary task
                        task = task_template.create_task(
                            cause=cause,
                            ssp_scenario=ssp_scenario,
                            dah_scenario=dah_scenario,
                            measure=measure,
                            hold_variable='None',
                            run_date=run_date,
                        )
                        tasks.append(task)
                else:
                    # Create the primary task
                    task = task_template.create_task(
                        cause=cause,
                        ssp_scenario=ssp_scenario,
                        dah_scenario='None',
                        measure=measure,
                        hold_variable='None',
                        run_date=run_date,
                    )
                    tasks.append(task)
dah_scenarios = ['Baseline']
if run_hold_variables:
    for cause in causes:
        for hold_variable in hold_variables[cause]:
            for ssp_scenario in rfc.ssp_scenarios:
                for measure in measures:
                    if cause == "malaria":
                        for dah_scenario in dah_scenarios:
                            # Create the primary task
                            task = task_template.create_task(
                                cause=cause,
                                ssp_scenario=ssp_scenario,
                                dah_scenario=dah_scenario,
                                measure=measure,
                                hold_variable=hold_variable,
                                run_date=run_date
                            )
                            tasks.append(task)
                    else:
                        # Create the primary task
                        task = task_template.create_task(
                            cause=cause,
                            ssp_scenario=ssp_scenario,
                            dah_scenario=None,
                            measure=measure,
                            hold_variable=hold_variable,
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
