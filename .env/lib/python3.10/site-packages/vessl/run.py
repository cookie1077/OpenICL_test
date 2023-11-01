import datetime
import time
from typing import List, Optional, TextIO

import yaml
from halo import Halo

from openapi_client import ExperimentYamlImportAPIInput, WorkspaceYamlImportAPIInput, ResponseRunExecutionListResponse, \
    ResponseRunExecutionInfo, InfluxdbWorkloadLog
from openapi_client.models import ResponseExperimentInfo
from vessl import __version__, vessl_api
from vessl.experiment import list_experiment_logs, read_experiment_by_id
from vessl.kernel_cluster import list_clusters
from vessl.organization import _get_organization_name
from vessl.project import _get_project_name
from vessl.util.constant import LOGO, WEB_HOST, colors
from vessl.workspace import read_workspace


def read_run(
    run_id: int,
    **kwargs,
) -> ResponseRunExecutionInfo:
    """Read run in the default organization/project. If you want to
    override the default organization/project, then pass `organization_name` or
    `project_name` as `**kwargs`.

    Args:
        run_id(int): run ID.

    Example:
        ```python
        vessl.read_run(
            run_id=123,
        )
        ```
    """
    return vessl_api.run_execution_read_api(
        execution_id=run_id,
        organization_name=_get_organization_name(**kwargs),
        project_name=_get_project_name(**kwargs),
    )


def update_run(
    run_id: int,
    description: str,
    **kwargs,
):
    """Update run in the default organization/project. If you want to
    override the default organization/project, then pass `organization_name` or
    `project_name` as `**kwargs`.

    Args:
        run_id(int): run ID.
        description(str): description of run to update.

    Example:
        ```python
        vessl.update_run(
            run_id=123,
            description="Update # of hidden layer 32->64",
        )
        ```
    """
    return vessl_api.run_execution_edit_api(
        execution_id=run_id,
        organization_name=_get_organization_name(**kwargs),
        project_name=_get_project_name(**kwargs),
        run_execution_edit_api_input={"message": description},
    )


def list_runs(
    statuses: List[str] = None,
    **kwargs,
) -> List[ResponseRunExecutionListResponse]:
    """List runs in the default organization/project. If you want to
    override the default organization/project, then pass `organization_name` or
    `project_name` as `**kwargs`.

    Args:
        statuses(List[str]): A list of status filter. Defaults to None.

    Example:
        ```python
        vessl.list_runs(
            statuses=["completed"]
        )
        ```
    """
    statuses = (
        [",".join(statuses)] if statuses else None
    )
    return vessl_api.run_execution_list_api(
        statuses=statuses,
        organization_name=_get_organization_name(**kwargs),
        project_name=_get_project_name(**kwargs),
    ).results


def list_run_logs(
    run_id: int,
    tail: int = 200,
    after: int = 0,
    **kwargs,
) -> List[InfluxdbWorkloadLog]:
    """List run logs in the default organization/project. If you want to
    override the default organization/project, then pass `organization_name` or
    `project_name` as `**kwargs`.

    Args:
        run_id (int): Run ID.
        tail (int): The number of lines to display from the end. Display all if
            -1. Defaults to 200.
        after (int): The number of starting lines to display from the start.
            Defaults to 0.

    Example:
        ```python
        vessl.list_run_logs(
            run_id=23,
        )
        ```
    """
    if tail == -1:
        tail = None

    return vessl_api.run_execution_logs_api(
        execution_id=run_id,
        log_limit=tail,
        organization_name=_get_organization_name(**kwargs),
        project_name=_get_project_name(**kwargs),
        log_start=after,
    ).logs


def create_run(
    yaml_file: TextIO,
    yaml_body: str,
    yaml_file_name: str,
    detached_mode = True,
    **kwargs,
):
    """Create run in the default organization/project. If you want to
    override the default organization/project, then pass `organization_name` or
    `project_name` as `**kwargs`.

    Args:
        yaml_file (TextIO): Run ID.
        yaml_body (str): YAML body.
        yaml_file_name (str): YAML file name.

    Example:
        ```python
        with open(file_name, "r") as yaml_file:
            vessl.create_run(
                yaml_file=yaml_file,
                yaml_body="",
                yaml_file_name=file_name,
            )
        ```
    """
    if yaml_body == "":
        body = yaml_file.read()
    else:
        body = yaml_body
    yaml_file_name = yaml_file_name.split("/")[-1]

    wrap_str(" Launch VESSL Run 👟", "green")
    organization = _get_organization_name(**kwargs)
    project = _get_project_name(**kwargs)
    wrap_str(f"   > Organization: {organization}", "cyan")
    wrap_str(f"   > Project: {project}", "cyan")

    spinner = Halo(text="Verifying YAML", text_color="cyan", spinner="dots", placement="right")
    spinner.start()
    interactive, out_str, yaml_obj = verify_yaml(body, spinner)

    if yaml_obj == False:
        spinner.stop_and_persist(
            symbol="😢", text=wrap_str(" YAML verification failed!", "red", do_print=False)
        )
        return
    else:
        spinner.stop_and_persist(
            symbol="✅", text=wrap_str(" YAML definition verified!", "green", do_print=False)
        )
    print(out_str)
    wrap_str(f" Running: {yaml_file_name} ➡️", "green")
    # yaml_obj["run"][0]["command"] = yaml_obj["run"][0]["command"].strip()
    clean_yaml_str = yaml.dump(yaml_obj, default_flow_style=False, sort_keys=False)
    msg_box(clean_yaml_str)

    spinner = Halo(
        text="Submitting Run definition to cluster ..",
        text_color="cyan",
        spinner="dots",
        placement="right",
    )
    spinner.start()

    response = vessl_api.run_spec_create_from_yamlapi(
        organization_name=_get_organization_name(**kwargs),
        project_name=_get_project_name(**kwargs),
        run_spec_create_from_yamlapi_input={"yaml_spec": clean_yaml_str},
    )

    spinner.stop_and_persist(
        symbol="✅", text=wrap_str(" Your Run is submitted to the cluster.", "green", do_print=False)
    )

    link = f"{WEB_HOST}/{_get_organization_name(**kwargs)}/{_get_project_name(**kwargs)}/runs/{response.run_execution.id}"
    hlink = f"\033]8;;{link}\033\\{link}\033]8;;\033\\"
    wrap_str(
        f" Check your Run at {hlink}",
        "cyan",
    )

    run_id = response.run_execution.id
    started = check_run_exec_started(response)
    if not started:
        return
    
    if detached_mode:
        log_link = f"{WEB_HOST}/{_get_organization_name(**kwargs)}/{_get_project_name(**kwargs)}/runs/{response.run_execution.id}/logs"
        h_log_link = f"\033]8;;{log_link}\033\\{log_link}\033]8;;\033\\"
        wrap_str(f" You can check the logs and progress of the run at {h_log_link}.", "cyan",)
        return 
    
    wrap_str(f" Showing Run logs from now !", "green")

    # fetch pod outputs
    run_finished_dt = None
    after = 0
    first_log = True
    while True:
        if (
                read_run(run_id=run_id).status not in ["pending", "running"]
                and run_finished_dt is None
        ):
            run_finished_dt = time.time()

        if run_finished_dt is not None and (time.time() - run_finished_dt) > 5:
            break

        logs = list_run_logs(
            run_id=run_id,
            before=int(time.time() - 5),
            after=after,
        )
        # do not print first log - generated while cluster was pending.
        if not first_log:
            print_logs(logs)
        else:
            first_log = False
        if len(logs) > 0:
            after = logs[-1].timestamp + 0.000001
        time.sleep(3)

def terminate_run(run_id: int, **kwargs):
    """Terminate run in the default organization/project. If you want to
        override the default organization/project, then pass `organization_name` or
        `project_name` as `**kwargs`.

        Args:
            run_id(int): Run ID.

        Example:
            ```python
            vessl.terminate_run(
                run_id=123,
            )
            ```
        """
    return vessl_api.run_execution_terminate_api(
        execution_id=run_id,
        organization_name=_get_organization_name(**kwargs),
        project_name=_get_project_name(**kwargs),
    )


def delete_run(run_id: int, **kwargs):
    """Delete run in the default organization/project. If you want to
    override the default organization/project, then pass `organization_name` or
    `project_name` as `**kwargs`.

    Args:
        run_id(int): Run ID.

    Example:
        ```python
        vessl.delete_experiment(
            run_id=123,
        )
        ```
    """
    return vessl_api.run_execution_delete_api(
        execution_id=run_id,
        organization_name=_get_organization_name(**kwargs),
        project_name=_get_project_name(**kwargs),
    )


def get_dt():
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dt = f"{colors.GREY}[{dt}]{colors.ENDC}"
    return dt


def wrap_str(string, color="default", end="", do_print=True):
    if color == "cyan":
        wrapped = f"{get_dt()}{colors.OKCYAN}{string}{end}{colors.ENDC}"
    elif color == "green":
        wrapped = f"{get_dt()}{colors.OKGREEN}{string}{end}{colors.ENDC}"
    elif color == "red":
        wrapped = f"{get_dt()}{colors.FAIL}{string}{end}{colors.ENDC}"
    elif color == "warn":
        wrapped = f"{get_dt()}{colors.WARNING}{string}{end}{colors.ENDC}"
    else:
        wrapped = f"{get_dt()}{string}{end}"
    if do_print:
        print(wrapped)
    else:
        return wrapped


def msg_box(msg):
    indent = 1
    lines = msg.split("\n")
    space = " " * indent
    width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    box += "".join([f"║{space}{line:<{width}}{space}║\n" for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)


def print_logs(logs: List[str]):
    timezone = datetime.datetime.now().astimezone().tzinfo
    for log in logs:
        ts = datetime.datetime.fromtimestamp(log.timestamp, tz=timezone).strftime("%H:%M:%S.%f")
        message = (
            log.message.replace("\\r", "\r")
            .replace("\\n", "\n")
            .replace("\\b", "\b")
            .replace("\\t", "\t")
            .replace("\\u001b", "\u001b")
        )
        for x in message.split("\n"):
            print(f"[{ts}] {x}")


# Check different stuffs in verify_yaml.
def verify_yaml(yaml_str, spinner):
    # replace \t to double spaces
    yaml_str = yaml_str.replace("\t", "  ")
    yaml_obj = yaml.safe_load(yaml_str)
    out_str = ""

    # Step 1: Check if all necessary keys exist.
    necessary_keys = [["image"], ["resources"]]
    for keyset in necessary_keys:
        _yaml = yaml_obj
        for key in keyset:
            if key not in _yaml.keys():
                wrap_str(f" Field {key} does not exist! Please specify them in your yaml.", "red"),
                return False, False, False
            _yaml = _yaml[key]

    # Check interactive
    is_interactive = True if "interactive" in yaml_obj.keys() else False

    # Check resources
    yaml_resource = yaml_obj["resources"]
    if "cluster" in yaml_resource:
        # Collect possible cluster and gpus
        cluster = yaml_resource["cluster"]
        cluster_cands = list_clusters()
        cluster_ids = dict()
        cluster_gpus = dict()
        for e in cluster_cands:
            cluster_ids[e.name] = e.id
            cluster_gpus[e.name] = e.available_gpus

        # Verify cluster
        if cluster not in cluster_ids.keys():
            wrap_str(
                f" {cluster} cluster does not exist! Please select among {list(cluster_ids.keys())}.",
                "red",
            )
            return False, False, False
        else:
            out_str += wrap_str(f"   ✓ Cluster verified", "cyan", "\n", do_print=False)

    if is_interactive:
        out_str += wrap_str("   ✓ Mode: Interactive", "cyan", do_print=False)
    else:
        out_str += wrap_str("   - 💡 Mode: Batch", "cyan", do_print=False)

    return is_interactive, out_str, yaml_obj


# Get yaml, verify yaml
def run_from_yaml(
    yaml_file: TextIO,
    yaml_body: str,
    yaml_file_name: str,
    **kwargs,
) -> ResponseExperimentInfo:
    if yaml_body == "":
        body = yaml_file.read()
    else:
        body = yaml_body
    yaml_file_name = yaml_file_name.split("/")[-1]

    wrap_str(" Launch VESSL Run 👟", "green")
    organization = _get_organization_name(**kwargs)
    project = _get_project_name(**kwargs)
    wrap_str(f"   > Organization: {organization}", "cyan")
    wrap_str(f"   > Project: {project}", "cyan")

    spinner = Halo(text="Verifying YAML", text_color="cyan", spinner="dots", placement="right")
    spinner.start()
    interactive, out_str, yaml_obj = verify_yaml(body, spinner)
    if yaml_obj == False:
        spinner.stop_and_persist(
            symbol="😢", text=wrap_str(" YAML verification failed!", "red", do_print=False)
        )
        return
    else:
        spinner.stop_and_persist(
            symbol="✅", text=wrap_str(" YAML definition verified!", "green", do_print=False)
        )
    print(out_str)
    wrap_str(f" Running: {yaml_file_name} ➡️", "green")
    yaml_obj["run"][0]["command"] = yaml_obj["run"][0]["command"].strip()
    clean_yaml_str = yaml.dump(yaml_obj)
    msg_box(clean_yaml_str)

    # workspace run
    if interactive:
        run_interactive_from_yaml(organization, body)
    else:
        run_batch_from_yaml(organization, project, body)
    return


# Applies yaml to cluster
def apply_yaml(organization, body, project=None, is_workspace=True):
    if is_workspace:
        workload = "workspace"
    else:
        workload = "experiment"
    spinner = Halo(
        text="Submitting Run definition to cluster ..",
        text_color="cyan",
        spinner="dots",
        placement="right",
    )
    spinner.start()
    if is_workspace:
        response = vessl_api.workspace_yaml_import_api(
            organization_name=organization,
            workspace_yaml_import_api_input=WorkspaceYamlImportAPIInput(
                data=body,
            ),
        )
    else:
        response = vessl_api.experiment_yaml_import_api(
            organization_name=organization,
            project_name=project,
            experiment_yaml_import_api_input=ExperimentYamlImportAPIInput(
                data=body,
            ),
        )
    spinner.stop_and_persist(
        symbol="✅", text=wrap_str(" Your Run is submitted to the cluster.", "green", do_print=False)
    )
    if is_workspace:
        link = f"{WEB_HOST}/{response.organization.name}/workspaces/{response.id}"
    else:
        link = f"{WEB_HOST}/{response.organization.name}/{response.project.name}/experiments/{response.number}"
    hlink = f"\033]8;;{link}\033\\{link}\033]8;;\033\\"
    wrap_str(
        f" Check your Run at {hlink}",
        "cyan",
    )
    return response

def check_run_exec_started(response):
    run_id = response.run_execution.id

    spinner = Halo(
        text="Cluster Pending ..",
        text_color="cyan",
        spinner="dots",
        placement="right",
    )
    spinner.start()
    not_started = True
    terminated = False
    while not_started and (not terminated):
        status = read_run(run_id=run_id).status
        if status != "pending":
            not_started = False
        if status in ["failed", "stopped"]:
            terminated = True

    if terminated:
        spinner.stop_and_persist(
            symbol="🏝️", text=wrap_str(f" Run terminated!", "green", do_print=False)
        )
        return False
    spinner.stop_and_persist(
        symbol="✅",
        text=wrap_str(f"> Your Run is assigned to the cluster.", "green", do_print=False),
    )

    spinner = Halo(
        text="Cluster Initializing .. ",
        text_color="cyan",
        spinner="dots",
        placement="right",
    )
    spinner.start()
    not_started = True
    while not_started and (not terminated):
        status = read_run(run_id=run_id).status
        if status == "running":
            not_started = False
        if status in ["failed", "stopped"]:
            terminated = True
    if terminated:
        spinner.stop_and_persist(
            symbol="🏝️", text=wrap_str(f" Run terminated!", "green", do_print=False)
        )
        return False
    spinner.stop_and_persist(
        symbol="✅", text=wrap_str(f"> Run has started!", "green", do_print=False)
    )
    print(LOGO)
    wrap_str(f" VESSL Run has succesfully launched! 🚀", "green")
    return True


# Check if workload have started
def check_started(response, is_workspace=True):
    if is_workspace:
        workload = "Workspace"
        workspace_id = response.id
    else:
        workload = "Experiment"
        experiment_id = response.id

    spinner = Halo(
        text="Cluster Pending ..",
        text_color="cyan",
        spinner="dots",
        placement="right",
    )
    spinner.start()
    not_started = True
    terminated = False
    while not_started and (not terminated):
        if is_workspace:
            status = read_workspace(workspace_id=workspace_id).status
        else:
            status = read_experiment_by_id(experiment_id).status
        if status != "pending":
            not_started = False
        if status in ["failed", "stopped"]:
            terminated = True

    if terminated:
        spinner.stop_and_persist(
            symbol="🏝️", text=wrap_str(f" {workload} terminated!", "green", do_print=False)
        )
        return False
    spinner.stop_and_persist(
        symbol="✅",
        text=wrap_str(f"> Your Run is assigned to the cluster.", "green", do_print=False),
    )

    spinner = Halo(
        text="Cluster Initializing .. ",
        text_color="cyan",
        spinner="dots",
        placement="right",
    )
    spinner.start()
    not_started = True
    while not_started and (not terminated):
        if is_workspace:
            status = read_workspace(workspace_id=workspace_id).status
        else:
            status = read_experiment_by_id(experiment_id).status
        if status == "running":
            not_started = False
        if status in ["failed", "stopped"]:
            terminated = True
    if terminated:
        spinner.stop_and_persist(
            symbol="🏝️", text=wrap_str(f" {workload} terminated!", "green", do_print=False)
        )
        return False
    spinner.stop_and_persist(
        symbol="✅", text=wrap_str(f"> Run has started!", "green", do_print=False)
    )
    print(LOGO)
    wrap_str(f" VESSL Run has succesfully launched! 🚀", "green")
    return True


def run_interactive_from_yaml(organization, body):
    response = apply_yaml(organization, body, is_workspace=True)
    started = check_started(response, is_workspace=True)
    if not started:
        return
    endpoints = response.endpoints.manually_defined_endpoints
    for endpoint in endpoints:
        wrap_str(f"📍 Endpoint {endpoint.name}: {endpoint.endpoint}")


def run_batch_from_yaml(organization, project, body):
    response = apply_yaml(organization, body, project, is_workspace=False)
    experiment_id = response.id
    experiment_number = response.number
    started = check_started(response, is_workspace=False)
    if not started:
        return
    wrap_str(f" Showing experiment logs from now !", "green")

    # fetch pod outputs
    experiment_finished_dt = None
    after = 0
    first_log = True
    while True:
        if (
            read_experiment_by_id(experiment_id).status not in ["pending", "running"]
            and experiment_finished_dt is None
        ):
            experiment_finished_dt = time.time()

        if experiment_finished_dt is not None and (time.time() - experiment_finished_dt) > 5:
            break

        # worker number: 0 since we do not handle distributed exps
        logs = list_experiment_logs(
            experiment_number=experiment_number,
            before=int(time.time() - 5),
            after=after,
            worker_numer=0,
        )
        # do not print first log - generated while cluster was pending.
        if not first_log:
            print_logs(logs)
        else:
            first_log = False
        if len(logs) > 0:
            after = logs[-1].timestamp + 0.000001
        time.sleep(3)
