"""
Common options for serving.
"""
import sys
from typing import Optional

import click

from openapi_client import ResponseModelServiceInfo
from vessl import vessl_api
from vessl.util.exception import VesslApiException


def serving_name_callback(
    ctx: click.Context, param: click.Parameter, serving_name: Optional[str]
) -> ResponseModelServiceInfo:
    if vessl_api.organization is None:
        vessl_api.set_organization()

    try:
        serving: ResponseModelServiceInfo = vessl_api.model_service_read_api(
            serving_name, vessl_api.organization.name
        )
    except VesslApiException as e:
        print(f"Invalid serving {serving_name}: {e.message}")
        sys.exit(1)

    return serving


serving_name_option = click.option(
    "--serving",
    type=click.STRING,
    required=True,
    callback=serving_name_callback,
    help="Name of serving.",
)

update_gateway_option = click.option(
    "--update-gateway/--no-update-gateway",
    "-g/-G",
    is_flag=True,
    default=False,
    help="Whether to update gateway so that it points to this revision.",
)

enable_gateway_if_off_option = click.option(
    "--enable-gateway-if-off/--no-enable-gateway-if-off",
    "-e/-E",
    is_flag=True,
    default=False,
    help="When updating gateway, whether to enable the gateway if it is currently off.",
)

update_gateway_weight_option = click.option(
    "--update-gateway-weight",
    type=click.INT,
    required=False,
    help=(
        "When updating gateway, the amount of traffic that should be "
        "directed to this revision (in percentage)."
    ),
)

update_gateway_port_option = click.option(
    "--update-gateway-port",
    type=click.INT,
    required=False,
    help=(
        "When updating gateway, the port to receive the traffic; "
        "this port must be defined in serving spec."
    ),
)

format_option = click.option(
    "--format",
    type=click.STRING,
    required=True,
    default="text",
    help=("Determines output format, supports text, json"),
)
