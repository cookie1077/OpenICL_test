from typing import List

from openapi_client import (
    ModelServiceGatewayUpdateAPIInput,
    ModelserviceModelServiceListResponse,
    ModelserviceModelServiceRevisionListResponse,
)
from openapi_client import OrmModelServiceGatewayTrafficSplitEntry as TrafficSplitEntry
from openapi_client import OrmAutoscalerConfig as AutoScalerConfig
from openapi_client import (
    ResponseModelServiceGatewayInfo,
    ResponseModelServiceInfo,
    ResponseModelServiceRevision,
    ResponseSimpleModelServiceRevision,
    ServingGatewayYamlImportAPIInput,
    ServingRevisionYamlImportAPIInput,
    ModelServiceRevisionUpdateAPIInput,
)
from vessl import vessl_api


def list_servings(organization: str) -> ModelserviceModelServiceListResponse:
    """Get a list of all servings in an organization

    Args:
        organization(str): The name of the organization.

    Example:
        ```python
        vessl.list_servings(organization="my-org")
        ```
    """
    return vessl_api.model_service_list_api(organization_name=organization)


def create_revision_from_yaml(
    organization: str, serving_name: str, yaml_body: str
) -> ResponseModelServiceRevision:
    """Create a new revision of serving from a YAML file.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.
        yaml_body(str): The YAML body of the serving.
            It is not deserialized YAML, but a whole yaml string.

    Example:
        ```python
        vessl.create_revision_from_yaml(
            organization="my-org",
            serving_name="my-serving",
            yaml_body=yaml_body)
        ```
    """
    payload = ServingRevisionYamlImportAPIInput(yaml_body)

    return vessl_api.serving_revision_yaml_import_api(
        organization_name=organization,
        model_service_name=serving_name,
        serving_revision_yaml_import_api_input=payload,
    )


def read_revision(
    organization: str, serving_name: str, revision_number: int
) -> ResponseModelServiceRevision:
    """Get a serving revision from a serving name and revision number.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.
        revision_number(int): The revision number of the serving.

    Example:
        ```python
        vessl.read_revision(
            organization="my-org",
            serving_name="my-serving",
            revision_number=1)
        ```
    """
    return vessl_api.model_service_revision_read_api(
        organization_name=organization,
        model_service_name=serving_name,
        revision_number=revision_number,
    )


def terminate_revision(organization: str, serving_name: str, revision_number: int):
    """
    Terminate a serving revision from a serving name and revision number.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.
        revision_number(int): The revision number of the serving.

    Example:
        ```python
        vessl.terminate_revision(
            organization="my-org",
            serving_name="my-serving",
            revision_number=1)
        ```
    """
    return vessl_api.model_service_revision_terminate_api(
        organization_name=organization,
        model_service_name=serving_name,
        revision_number=revision_number,
    )


def update_revision_autoscaler_config(
    organization: str,
    serving_name: str,
    revision_number: int,
    auto_scaler_config: AutoScalerConfig,
):
    """
    Update the autoscaler config of a serving revision from a serving name and revision number.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.
        revision_number(int): The revision number of the serving.
        auto_scaler_config(AutoScalerConfig): The autoscaler config of the serving.

    Example:
        ```python
        vessl.update_revision_autoscaler_config(
            organization="my-org",
            serving_name="my-serving",
            revision_number=1,
            auto_scaler_config=AutoScalerConfig(
                min_replicas=1,
                max_replicas=2,
                target_cpu_utilization_percentage=80,
            ))
        ```
    """
    return vessl_api.model_service_revision_update_api(
        organization_name=organization,
        model_service_name=serving_name,
        revision_number=revision_number,
        model_service_revision_update_api_input=ModelServiceRevisionUpdateAPIInput(
            auto_scaler_config=auto_scaler_config,
        ),
    )


def list_revisions(
    organization: str, serving_name: str
) -> List[ResponseSimpleModelServiceRevision]:
    """Get a list of all revisions of a serving.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.

    Examples:
        ```python
        vessl.list_revisions(
            organization="my-org",
            serving_name="my-serving")
        ```
    """
    resp: ModelserviceModelServiceRevisionListResponse = vessl_api.model_service_revision_list_api(
        organization_name=organization,
        model_service_name=serving_name,
    )
    return resp.results


def read_gateway(organization: str, serving_name: str) -> ResponseModelServiceGatewayInfo:
    """Get the gateway of a serving.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.

    Examples:
        ```python
        vessl.read_gateway(
            organization="my-org",
            serving_name="my-serving")
        ```
    """
    model_service: ResponseModelServiceInfo = vessl_api.model_service_read_api(
        model_service_name=serving_name,
        organization_name=organization,
    )
    return model_service.gateway_config


def update_gateway(
    organization: str, serving_name: str, gateway: ModelServiceGatewayUpdateAPIInput
) -> ResponseModelServiceGatewayInfo:
    """Update the gateway of a serving.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.
        gateway(ModelServiceGatewayUpdateAPIInput): The gateway of the serving.

    Examples:
        ```python
        from openapi_client import ModelServiceGatewayUpdateAPIInput
        from openapi_client import OrmModelServiceGatewayTrafficSplitEntry

        gateway = ModelServiceGatewayUpdateAPIInput(
            enabled=True,
            ingress_host="my-endpoint",
            traffic_split=[
                OrmModelServiceGatewayTrafficSplitEntry(
                    revision_number=1,
                    port=2222,
                    traffic_weight=100,
                )
            ],
        )

        vessl.update_gateway(
            organization="my-org",
            serving_name="my-serving",
            gateway=gateway)
        ```
    """
    return vessl_api.model_service_gateway_update_api(
        model_service_name=serving_name,
        organization_name=organization,
        model_service_gateway_update_api_input=gateway,
    )


def _get_updated_traffic_split_rule(
    rules_current: List[TrafficSplitEntry], revision_number: int, weight: int, port: int
) -> List[TrafficSplitEntry]:
    """
    Combines the previous traffic split rule with new rule.
    When filling the remaining weight, this function uses the one with higher revision number.

    For example, with the current rule of:
    - revision #2 (port 2222) 70%
    - revision #3 (port 3333) 30%

    with a call to this function with:
    - revision #4 (port 4444) 50%

    yields a new rule of:
    - revision #4 (port 4444) 50%
    - revision #3 (port 3333) 30%
    - revision #2 (port 2222) 20%

    Revision #3 takes priority over #2, because it has the higher number (3 > 2).
    """
    # Sort from latest revision (with highest number) to oldest
    rules_current = sorted(rules_current, key=lambda x: x.revision_number, reverse=True)

    rules_new: List[TrafficSplitEntry] = [
        TrafficSplitEntry(revision_number=revision_number, port=port, traffic_weight=weight)
    ]

    weight_remaining = 100 - weight

    # Iterate through current traffic rules and add them if possible
    for rule in rules_current:
        if weight_remaining <= 0:
            break
        new_weight = min(weight_remaining, rule.traffic_weight)
        rules_new.append(
            TrafficSplitEntry(
                revision_number=rule.revision_number, port=rule.port, traffic_weight=new_weight
            )
        )
        weight_remaining -= new_weight
        if weight_remaining <= 0:
            break

    if weight_remaining > 0:
        # This can happen if rules_current's weight do not sum up to 100
        # (this is possible for disabled gateways).
        # Handle this case safely by delegating all remaining weights to our target rule.
        rules_new[0].traffic_weight += weight_remaining

    return rules_new


def update_gateway_for_revision(
    organization: str,
    serving_name: str,
    revision_number: int,
    port: int,
    weight: int,
) -> ResponseModelServiceGatewayInfo:
    """Update the current gateway of a serving for a specific revision.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.
        revision_number(int): The revision number of the serving.
        port(int): The port of the revision will use for gateway.
        weight(int): The weight of the traffic will be distributed to revision_number.

    Examples:
        ```python
        vessl.update_gateway_for_revision(
            organization="my-org",
            serving_name="my-serving",
            revision_number=1,
            port=2222,
            weight=100)
        ```
    """
    gateway_current = read_gateway(organization=organization, serving_name=serving_name)

    rules_new = _get_updated_traffic_split_rule(
        rules_current=gateway_current.rules or [],
        revision_number=revision_number,
        port=port,
        weight=weight,
    )

    gateway_updated = vessl_api.model_service_gateway_update_api(
        organization_name=organization,
        model_service_name=serving_name,
        model_service_gateway_update_api_input=ModelServiceGatewayUpdateAPIInput(
            enabled=True,
            ingress_host=gateway_current.endpoint,
            ingress_class=gateway_current.ingress_class,
            annotations=gateway_current.annotations,
            traffic_split=rules_new,
        ),
    )
    return gateway_updated


def update_gateway_from_yaml(
    organization: str, serving_name: str, yaml_body: str
) -> ResponseModelServiceGatewayInfo:
    """Update the gateway of a serving from a YAML file.

    Args:
        organization(str): The name of the organization.
        serving_name(str): The name of the serving.
        yaml_body(str): The YAML body of the serving.
            It is not deserialized YAML, but a whole yaml string

    Examples:
        ```python
        vessl.update_gateway_from_yaml(
            organization="my-org",
            serving_name="my-serving",
            yaml_body=yaml_body)
        ```
    """
    payload = ServingGatewayYamlImportAPIInput(yaml_body)

    return vessl_api.serving_gateway_yaml_import_api(
        organization_name=organization,
        model_service_name=serving_name,
        serving_gateway_yaml_import_api_input=payload,
    )
