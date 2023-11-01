"""
Serving command definition and delegation.
"""
import sys
from typing import List, Optional, TextIO

import click
import yaml
import json

from openapi_client.models import ResponseModelServiceInfo
from openapi_client.models.response_kernel_cluster import ResponseKernelCluster
from vessl import vessl_api
from vessl.cli._base import VesslCommand, VesslGroup
from vessl.cli.serve.util import (
    list_http_ports,
    print_gateway,
    print_revision,
    validate_revision_yaml_obj,
)
from vessl.serving import (
    create_revision_from_yaml,
    list_revisions,
    list_servings,
    read_gateway,
    read_revision,
    update_gateway_for_revision,
    update_gateway_from_yaml,
    terminate_revision,
    update_revision_autoscaler_config,
)
from vessl.util.exception import InvalidYAMLError, VesslApiException

from .command_options import (
    enable_gateway_if_off_option,
    serving_name_option,
    update_gateway_option,
    update_gateway_port_option,
    update_gateway_weight_option,
    format_option,
)

cli = VesslGroup("serve")


@cli.command("list", cls=VesslCommand)
def serving_list():
    """
    List servings in current organization.
    """
    servings: List[ResponseModelServiceInfo] = list_servings(
        organization=vessl_api.organization.name
    ).results

    print(f"{len(servings)} serving(s) found.\n")
    for serving in servings:
        kernel_cluster: ResponseKernelCluster = serving.kernel_cluster
        status: str = serving.status  # "ready", "running", "error"

        print(f"{serving.name} (cluster {kernel_cluster.name}): status {status.capitalize()}")


@cli.group("revision")
def cli_revision():
    """
    Root command for revision-related commands.
    """
    pass


@cli_revision.command("create", cls=VesslCommand)
@serving_name_option
@click.option(
    "-f",
    "--file",
    type=click.File("r"),
    required=True,
    help="Path to YAML file for serving revision definition.",
)
@update_gateway_option
@enable_gateway_if_off_option
@update_gateway_weight_option
@update_gateway_port_option
def revision_create_with_yaml(
    serving: ResponseModelServiceInfo,
    file: TextIO,
    update_gateway: bool,
    enable_gateway_if_off: bool,
    update_gateway_weight: Optional[int] = None,
    update_gateway_port: Optional[int] = None,
):
    """
    Create serving revision with spec written in YAML.
    """
    if not update_gateway and (
        update_gateway_weight is not None or update_gateway_port is not None
    ):
        print("Cannot specify traffic weight or port when not updating gateway.")
        sys.exit(1)

    yaml_body = file.read()
    try:
        yaml_obj = yaml.safe_load(yaml_body)
    except yaml.YAMLError as e:
        print(f"Error: invalid YAML\n{e}")
        sys.exit(1)

    try:
        validate_revision_yaml_obj(yaml_obj=yaml_obj)
    except InvalidYAMLError as e:
        print(f"Error: invalid YAML: {e.message}")
        sys.exit(1)

    if update_gateway:
        # Do as much validation as possible before actually creating revision.
        # weight check
        if update_gateway_weight is None:
            update_gateway_weight = 100
        elif not 1 <= update_gateway_weight <= 100:
            print(f"Error: invalid weight: {update_gateway_weight}% (must be between 1 and 100)")
            sys.exit(1)

        # port check
        http_ports = list_http_ports(yaml_obj=yaml_obj)
        if update_gateway_port is None:
            if len(http_ports) != 1:
                print(
                    "Error: port for gateway was not specified, and could not automatically "
                    "determine which port to use.\n"
                    f"{len(http_ports)} port(s) was found: " + ", ".join(map(str, http_ports))
                )
                sys.exit(1)
            update_gateway_port = http_ports[0]
            print(f"Automatically choosing port {update_gateway_port} for gateway.")
        elif not 1 <= update_gateway_port <= 65535:
            print(f"Error: invalid port: {update_gateway_port}")
            sys.exit(1)
        else:
            if update_gateway_port not in http_ports:
                print(
                    f"Error: user requested port {update_gateway_port} for gateway, "
                    f"but the port was not found in revision's HTTP exposes "
                    f"({', '.join(map(str, http_ports))})."
                )
                sys.exit(1)

        # gateway status check
        if not enable_gateway_if_off:
            gateway_current = read_gateway(
                organization=vessl_api.organization.name, serving_name=serving.name
            )
            if not gateway_current.enabled:
                print("Cannot update gateway because it is not enabled. Please enable it first.")
                print("NOTE (current status of gateway):")
                print_gateway(gateway_current)
                sys.exit(1)

    revision = create_revision_from_yaml(
        organization=vessl_api.organization.name, serving_name=serving.name, yaml_body=yaml_body
    )
    print(f"Successfully created revision in serving {serving.name}.\n")
    print_revision(revision)

    if update_gateway:
        gateway_updated = update_gateway_for_revision(
            vessl_api.organization.name,
            serving_name=serving.name,
            revision_number=revision.number,
            port=update_gateway_port,
            weight=update_gateway_weight,
        )
        print(f"Successfully updated gateway for revision #{revision.number}.\n")
        print_gateway(gateway_updated)
    else:
        print(
            "NOTE: Since --update-gateway option was not given, "
            "you cannot currently access this revision via gateway.\n\n"
            "Either use --update-gateway when creating revision, or update gateway manually."
        )


@cli_revision.command("show", cls=VesslCommand)
@serving_name_option
@click.option("--number", "-n", required=True, type=int, help="Number of revision.")
@format_option
def revision_show(serving: ResponseModelServiceInfo, number: int, format: str):
    """
    Show current status and information about a serving revision.
    """
    try:
        revision = read_revision(
            organization=vessl_api.organization.name,
            serving_name=serving.name,
            revision_number=number,
        )
    except VesslApiException as e:
        print(f"Failed to read revision #{number} of serving {serving.name}: {e.message}")
        sys.exit(1)

    if format == "json":
        print(json.dumps(revision.to_dict(), default=str))
    else:
        print_revision(revision, verbose=True)


@cli_revision.command("list", cls=VesslCommand)
@serving_name_option
@format_option
def revision_list(serving: ResponseModelServiceInfo, format: str):
    """
    List all revisions.
    """
    try:
        revisions = list_revisions(
            organization=vessl_api.organization.name,
            serving_name=serving.name,
        )
    except VesslApiException as e:
        print(f"Failed to list revisions of serving {serving.name}: {e.message}")
        sys.exit(1)

    if format == "json":
        print(json.dumps([r.to_dict() for r in revisions], default=str))
    else:
        print(f"{len(revisions)} revision(s) found.\n")
        for i, revision in enumerate(revisions):
            if i > 0:
                print()
            print_revision(revision)


@cli_revision.command("terminate", cls=VesslCommand)
@serving_name_option
@click.option("--number", "-n", required=True, type=int, help="Number of revision.")
def revision_terminate(serving: ResponseModelServiceInfo, number: int):
    """
    Terminate specified revision.
    """
    try:
        terminate_revision(
            organization=vessl_api.organization.name,
            serving_name=serving.name,
            revision_number=number,
        )
        print("Successfully terminated revision.")
    except VesslApiException as e:
        print(f"Failed to terminate revision #{number} of serving {serving.name}: {e.message}")
        sys.exit(1)


@cli_revision.command("update-autoscaler-config", cls=VesslCommand)
@serving_name_option
@click.option("--number", "-n", required=True, type=int, help="Number of revision.")
@click.option("--min-replicas", required=False, type=int, help="Number of min replicas.")
@click.option("--max-replicas", required=False, type=int, help="Number of min replicas.")
def revision_update_autoscaler_config(
    serving: ResponseModelServiceInfo,
    number: int,
    min_replicas: Optional[int],
    max_replicas: Optional[int],
):
    """
    Update revision's autoscaler config
    """
    try:
        current_revision = read_revision(
            organization=vessl_api.organization.name,
            serving_name=serving.name,
            revision_number=number,
        )

        conf = current_revision.autoscaler_config
        if min_replicas is not None:
            conf.min_replicas = min_replicas
        if max_replicas is not None:
            conf.max_replicas = max_replicas

        update_revision_autoscaler_config(
            organization=vessl_api.organization.name,
            serving_name=serving.name,
            revision_number=number,
            auto_scaler_config=conf,
        )
        print("Successfully updated autoscaler config.")
    except VesslApiException as e:
        print(
            f"Failed to update autoscaler config of revision #{number} of serving {serving.name}: {e.message}"
        )
        sys.exit(1)


@cli.group("gateway")
def cli_gateway():
    """
    Root command for gateway-related commands.
    """
    pass


@cli_gateway.command("show", cls=VesslCommand)
@serving_name_option
@format_option
def gateway_show(serving: ResponseModelServiceInfo, format: str):
    """
    Show current status of the gateway of a serving.
    """
    try:
        gateway = read_gateway(
            organization=vessl_api.organization.name,
            serving_name=serving.name,
        )
    except VesslApiException as e:
        print(f"Failed to read gateway of serving {serving.name}: {e.message}")
        sys.exit(1)

    if format == "json":
        print(json.dumps(gateway.to_dict(), default=str))
    else:
        print_gateway(gateway)


@cli_gateway.command("update", cls=VesslCommand)
@serving_name_option
@click.option(
    "-f",
    "--file",
    type=click.File("r"),
    required=True,
    help="Path to YAML file for serving revision definition.",
)
def gateway_update_with_yaml(serving: ResponseModelServiceInfo, file: TextIO):
    """
    Update serving gateway with spec written in YAML.
    """
    yaml_body = file.read()

    gateway = update_gateway_from_yaml(
        organization=vessl_api.organization.name, serving_name=serving.name, yaml_body=yaml_body
    )
    print(f"Successfully update gateway of serving {serving.name}.\n")
    print_gateway(gateway)
