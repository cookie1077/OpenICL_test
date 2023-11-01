from typing import List

from kubernetes.config import KUBE_CONFIG_DEFAULT_LOCATION

from vessl.util.constant import VESSL_HELM_CHART_NAME, API_HOST, VESSL_ENV
from vessl.util.exception import VesslRuntimeException


class ClusterInstallCommandBuilder:
    def __init__(self) -> None:
        self.command_template = ("cat <<EOF | helm install vessl vessl/{helm_chart_name} "
                                 "--create-namespace --namespace {namespace} --kubeconfig {kubeconfig_location} "
                                 "{extra_values}"
                                 "-f -\n{values_yaml}\nEOF")
        self.env = VESSL_ENV
        self.api_server = API_HOST
        self.namespace = "vessl"
        self.kubeconfig = KUBE_CONFIG_DEFAULT_LOCATION
        self.cluster_name = ""
        self.access_token = ""
        self.provider_type = "on-premise"
        self.local_path_provisioner_enabled = True
        self.longhorn_enabled = True

        # optional) As these overrides default values.yaml, should check if this is empty
        # and skip when empty
        self.helm_values = []
        self.pod_resource_path = ""

    def with_namespace(self, namespace: str) -> "ClusterInstallCommandBuilder":
        self.namespace = namespace
        return self

    def with_kubeconfig(self, kubeconfig_path: str) -> "ClusterInstallCommandBuilder":
        self.kubeconfig = kubeconfig_path
        return self

    def with_cluster_name(self, cluster_name: str) -> "ClusterInstallCommandBuilder":
        self.cluster_name = cluster_name
        return self

    def with_access_token(self, access_token: str) -> "ClusterInstallCommandBuilder":
        self.access_token = access_token
        return self

    def with_provider_type(self, provider_type: str) -> "ClusterInstallCommandBuilder":
        self.provider_type = provider_type
        return self

    def with_local_path_provisioner_enabled(self, enabled) -> "ClusterInstallCommandBuilder":
        """
        - local-path-provisioner.enabled={enabled}
        """
        self.local_path_provisioner_enabled = enabled
        return self

    def with_longhorn_enabled(self, enabled: bool) -> "ClusterInstallCommandBuilder":
        """
        - longhorn.enabled={enabled}
        """
        self.longhorn_enabled = enabled
        return self

    def with_helm_values(self, helm_values: List[str]) -> "ClusterInstallCommandBuilder":
        self.helm_values = helm_values
        return self

    # @XXX(seokju) rough assumption: k0s user will install k0s nodes
    # if this becomes a problem, patch chart and cli.
    def with_pod_resource_path(self, path: str) -> "ClusterInstallCommandBuilder":
        """
        - dcgm-exporter.kubeletPath={path}
        """
        self.pod_resource_path = path
        return self

    def build(self) -> str:
        if not self.cluster_name:
            raise VesslRuntimeException("cluster_name is required to build install command")
        if not self.access_token:
            raise VesslRuntimeException("access_token is required to build install command")

        yaml_string = f"""
agent:
  env: {self.env}
  apiServer: {self.api_server}
  clusterName: {self.cluster_name}
  accessToken: {self.access_token}
  providerType: {self.provider_type}
local-path-provisioner:
  enabled: {'true' if self.local_path_provisioner_enabled else 'false'}
longhorn:
  enabled: {'true' if self.longhorn_enabled else 'false'}
"""
        if self.pod_resource_path:
            yaml_string += f"""
dcgm-exporter:
  kubeletPath: {self.pod_resource_path}
"""
        if self.env != "prod":
            yaml_string += f"""
prometheus-remote-write:
  server:
    remoteWrite:
      - name: "vessl-remote-write"
        url: "https://remote-write-gateway.dev.vssl.ai/remote-write"
        authorization:
          type: "Token"
          credentials_file: "/etc/secrets/token"
        write_relabel_configs:
          - action: "labeldrop"
            regex: "feature_node_kubernetes_io_(.+)"
          - action: "labeldrop"
            regex: "label_feature_node_kubernetes_io_(.+)"
          - action: "labeldrop"
            regex: "minikube_(.+)"
"""
        extra_values = ""
        for helm_value in self.helm_values:
            if helm_value.strip():
                extra_values += f"--set {helm_value.strip()} "

        return self.command_template.format(
            helm_chart_name=VESSL_HELM_CHART_NAME,
            namespace=self.namespace,
            kubeconfig_location=self.kubeconfig,
            extra_values=extra_values,
            values_yaml=yaml_string,
        )
