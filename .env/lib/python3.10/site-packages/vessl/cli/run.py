import os
import time

import click

from vessl.cli._base import VesslGroup, vessl_argument
from vessl.cli._util import print_table, truncate_datetime, print_data, prompt_text, print_logs, print_info, \
    print_success, Endpoint
from vessl.cli.organization import organization_name_option
from vessl.cli.project import project_name_option
from vessl.run import (
    list_runs, read_run, update_run, list_run_logs, delete_run, create_run, wrap_str, terminate_run,
)


def prompt_run_description(ctx: click.Context, param: click.Parameter, description: str):
    return prompt_text("Description to change", False)


@click.command(name="run", cls=VesslGroup)
def cli():
    pass


@cli.vessl_command()
@vessl_argument(
    "run_id",
    type=click.INT,
    required=True,
)
@organization_name_option
@project_name_option
def read(run_id: int):
    run = read_run(run_id=run_id)

    resource = "None"
    if run.run_spec.kernel_resource_spec_from_preset:
        run_resource_spec = run.run_spec.kernel_resource_spec_from_preset
        resource = {
            "Name": run_resource_spec.name,
            "CPU Type": run_resource_spec.cpu_type,
            "CPU Limit": run_resource_spec.cpu_limit,
            "Memory Limit": run_resource_spec.memory_limit,
            "GPU Type": run_resource_spec.gpu_type,
            "GPU Limit": run_resource_spec.gpu_limit,
        }

    image = "None"
    if run.run_spec.kernel_image:
        run_kernel_image = run.run_spec.kernel_image
        image = {
            "Name": run_kernel_image.name,
            "URL": run_kernel_image.image_url,
        }

    commands = []
    if run.run_spec.spec.command and len(run.run_spec.spec.command) != 0:
        for c in run.run_spec.spec.command:
            if c.wait:
                commands.append({
                    "wait": c.wait
                })

            else:
                if c.workdir:
                    commands.append({
                        "workdir": c.workdir,
                        "command": c.command
                    })
                else:
                    commands.append({
                        "command": c.command
                    })

    interactive = False
    if run.run_spec.spec.interactive:
        interactive = {
            "max_runtime": run.run_spec.spec.interactive.max_runtime,
            "jupyter_idle_timeout": run.run_spec.spec.interactive.jupyter.idle_timeout,
        }

    ports = []
    if run.run_spec.spec.ports and len(run.run_spec.spec.ports) != 0:
        run_ports = run.run_spec.spec.ports
        for p in run_ports:
            ports.append({
                "name": p.name,
                "number": p.target_port,
                "protocol": p.protocol,
            })

    env_vars = []
    if run.run_spec.spec.env_vars and len(run.run_spec.spec.env_vars):
        run_env_vars = run.run_spec.spec.env_vars
        for v in run_env_vars:
            env_vars.append({
                "key": v.key,
                "value": v.default,
                "secret": v.secret,
            })

    service_account = "None"
    if run.run_spec.spec.service_account_name != "":
        service_account = run.run_spec.spec.service_account_name

    print_data(
        {
            "ID": run.id,
            "Name": run.run_spec.title,
            "Description": run.run_spec.description,
            "Status": run.status,
            "Status reason": run.status_reason,
            "Created": truncate_datetime(run.create_dt),
            "Cluster": run.run_spec.cluster.name,
            "Resource": resource,
            "Image": image,
            "Start commands": commands,
            "Interactive": interactive,
            "Environment variables": env_vars,
            "Ports": ports,
            "Service account": service_account,
            "Termination protection": run.run_spec.spec.termination_protection,
        }
    )


@cli.vessl_command()
@organization_name_option
@project_name_option
def list():
    runs = list_runs()
    print_table(
        runs,
        ["ID", "Name", "Type", "Status", "Created", "Description"],
        lambda x: [
            x.id,
            x.run_spec.title,
            x.run_spec.type,
            x.status,
            truncate_datetime(x.create_dt),
            x.message,
        ]
    )

@cli.vessl_command()
@organization_name_option
@project_name_option
@click.option(
    "-f",
    "--file",
    type=click.STRING,
    help="yaml file for run definition.",
)
@click.option(
    "-d",
    "--detached",
    default=True,
    type=click.BOOL,
    help="create run without logs",
)
def create(file: str, detached: bool):
    if not file:
        print("Missing argument")
        return
    if not os.path.exists(file):
        print(wrap_str(f" Yaml does not exist! Check yaml path {file}.", "red"))
        return
    with open(file, "r") as yaml_file:
        create_run(yaml_file=yaml_file, yaml_body="", yaml_file_name=file, detached_mode=detached)


@cli.vessl_command()
@vessl_argument(
    "run_id",
    type=click.INT,
    required=True,
)
@vessl_argument(
    "description",
    type=click.STRING,
    required=True,
    prompter=prompt_run_description
)
@organization_name_option
@project_name_option
def update(run_id: int, description: str):
    update_run(run_id, description)
    print_success(f"Updated #{run_id}.\n")


@cli.vessl_command()
@vessl_argument(
    "run_id",
    type=click.INT,
    required=True,
)
@click.option(
    "--tail",
    type=click.INT,
    default=200,
    help="Number of lines to display (from the end).",
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    default=False,
)
@organization_name_option
@project_name_option
def logs(run_id: int, tail: int, follow: bool):
    if not follow:
        logs = list_run_logs(run_id=run_id, tail=tail)
        print_logs(logs)
        print_info(f"Displayed last {len(logs)} lines of #{run_id}.")
        return

    after = 0
    run_finished_dt = None
    while True:
        if (
            read_run(run_id).status not in ["pending", "running"]
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
        print_logs(logs)
        if len(logs) > 0:
            after = logs[-1].timestamp + 0.000001

        time.sleep(3)


@cli.vessl_command()
@vessl_argument(
    "run_id",
    type=click.INT,
    required=True,
)
@organization_name_option
@project_name_option
def terminate(run_id: int):
    run = terminate_run(run_id=run_id)
    print_success(
        f"Terminated '#{run.id}'.\n"
    )

@cli.vessl_command()
@vessl_argument(
    "run_id",
    type=click.INT,
    required=True,
)
@organization_name_option
@project_name_option
def delete(run_id: int):
    delete_run(run_id)
    print_success(f"Deleted #{run_id}.\n")