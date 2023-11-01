import sys

import click

import vessl
from vessl.cli._base import VesslGroup, vessl_argument
from vessl.cli._util import (
    Endpoint,
    generic_prompter,
    print_data,
    print_error,
    print_info,
    print_success,
    print_table,
    prompt_choices,
)
from vessl.organization import (
    create_organization,
    list_organizations,
    read_organization,
)
from vessl.util.exception import InvalidOrganizationError


def organization_name_prompter(ctx: click.Context, param: click.Parameter, value: str) -> str:
    organizations = list_organizations()
    return prompt_choices("Organization", [x.name for x in organizations])


@click.command(name="organization", cls=VesslGroup)
def cli():
    pass


@cli.vessl_command()
@vessl_argument(
    "name",
    type=click.STRING,
    required=True,
    prompter=organization_name_prompter,
)
def read(name: str):
    organization = read_organization(organization_name=name)
    print_data(
        {
            "Name": organization.name,
            "Description": organization.description,
        }
    )


@cli.vessl_command()
def list():
    organizations = list_organizations()
    print_table(
        organizations,
        ["Name"],
        lambda x: [x.name],
    )


@cli.vessl_command()
@vessl_argument(
    "name",
    type=click.STRING,
    required=True,
    prompter=generic_prompter("Organization name"),
)
def create(name: str):
    organization = create_organization(organization_name=name)
    print_success(
        f"Created '{organization.name}'.\n"
        f"For more info: {Endpoint.organization.format(organization.name)}"
    )


def organization_name_callback(ctx: click.Context, param: click.Parameter, value: str) -> str:
    try:
        organization_name = vessl.vessl_api.set_organization(value)
        print_info(f"Organization: {organization_name}")
    except InvalidOrganizationError:
        print_error("Invalid organization. Please choose an organization using `vessl configure`.")
        sys.exit(1)


# Ensure this is called before other options with `is_eager=True` for
# other callbacks that need organization to be preconfigured.
organization_name_option = click.option(
    "--organization",
    "organization_name",
    type=click.STRING,
    hidden=True,
    is_eager=True,
    expose_value=False,
    callback=organization_name_callback,
    help="Override default organization.",
)
