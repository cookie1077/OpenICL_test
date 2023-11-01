from typing import List

from openapi_client import OrmModelServiceGatewayTrafficSplitEntry as TrafficSplitEntry
from openapi_client import (
    ResponseKernelImage,
    ResponseModelServiceGatewayInfo,
    ResponseModelServiceRevision,
)
from openapi_client.models import OrmAutoscalerConfig, OrmWorkloadPort
from vessl.cli._util import print_data
from vessl.util.exception import InvalidYAMLError


def print_revision(revision: ResponseModelServiceRevision, verbose: bool = False):
    if not verbose:
        data = {
            "Number": revision.number,
            "Status": revision.status,
            "Message": revision.message,
        }
    else:
        kernel_image: ResponseKernelImage = revision.kernel_image
        autoscaler_config: OrmAutoscalerConfig = revision.autoscaler_config
        ports: List[OrmWorkloadPort] = revision.ports.ports
        data = {
            "Number": revision.number,
            "Status": revision.status,
            "Message": revision.message,
            "Available Replicas": revision.available_replicas,
            "Deployment Spec": {
                "Image URL": kernel_image.image_url,
                "Ports": [
                    {"Name": port.name, "Port": port.port, "Type": port.expose_type}
                    for port in ports
                ],
            },
            "Autoscaler Config": {
                "Min Replicas": autoscaler_config.min_replicas,
                "Max Replicas": autoscaler_config.min_replicas,
                "Resource": autoscaler_config.resource_target,
                "Target": autoscaler_config.average_utilization,
            },
        }
    print_data(data)


def print_gateway(gateway: ResponseModelServiceGatewayInfo):
    def _prettify_traffic_split_entry(entry: TrafficSplitEntry) -> str:
        """
        Returns a pretty representation of given traffic split entry.

        Examples:
        - "########   80%:   3 (port 8000)"
        - "##         20%:  12 (port 3333)"
        """
        gauge_char = "#"
        gauge_width = (entry.traffic_weight + 9) // 10  # round up (e.g. 31%-40% -> 4)
        gauge = gauge_char * gauge_width

        return (
            f"{gauge: <10} {entry.traffic_weight: <3}%: {entry.revision_number: >3} "
            f"(port {entry.port})"
        )

    print_data(
        {
            "Enabled": gateway.enabled,
            "Status": gateway.status,
            "Endpoint": gateway.endpoint or "(not set)",
            "Ingress Class": gateway.ingress_class or "(empty)",
            "Annotations": gateway.annotations or "(empty)",
            "Traffic Targets": (
                list(map(_prettify_traffic_split_entry, gateway.rules))
                if gateway.rules
                else "(empty)"
            ),
        }
    )


def validate_revision_yaml_obj(yaml_obj):
    """
    Client-side validation. We do not need a full validation here, as it will
    be done on the server side anyway. Only perform necessary validations to
    ensure that other client-side logics run safely.

    Raises:
        InvalidYAMLError: this YAML object for revision is invalid.
    """

    def _check_type(obj, type_: type):
        assert isinstance(obj, type_)
        return obj

    try:
        if "ports" in yaml_obj:
            ports: list = _check_type(yaml_obj["ports"], list)

            for port in ports:
                expose_type = _check_type(port["type"], str)
                assert expose_type in ["http", "tcp"]

                port_number = _check_type(port["port"], int)
                assert 1 <= port_number <= 65535

    except (KeyError, AssertionError) as e:
        raise InvalidYAMLError(message=str(e))


def list_http_ports(yaml_obj) -> List[int]:
    """
    Lists all HTTP expose ports of the revision.

    This function assumes the YAML object is valid (i.e. has valid port definitions.)
    """

    if "ports" not in yaml_obj:
        return []

    http_ports: List[int] = []

    try:
        ports: list = yaml_obj["ports"]
        for port in ports:
            expose_type: str = port["type"]
            if expose_type == "http":
                port_number: int = port["port"]
                http_ports.append(port_number)

    except (KeyError, AssertionError) as e:
        raise InvalidYAMLError(message=str(e))

    return http_ports
