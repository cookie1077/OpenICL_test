import os
from pathlib import Path

import toml
from schema import And, Optional, Schema

DEFAULT_VESSL_DIR = os.path.join(str(Path.home()), ".vessl")
DEFAULT_CONFIG_PATH = os.environ.get("VESSL_CONFIG_PATH", os.path.join(DEFAULT_VESSL_DIR, "config"))
SAVE_CONFIG = os.environ.get("VESSL_SAVE_CONFIG", "true").lower() == "true"

class ConfigLoader:
    def __init__(self, filename: str, schema: Schema):
        self.filename = filename
        self.schema = schema
        self.config = {}
        self._load()  # May raise TomlDecodeError
        self._validate()  # May raise SchemaError

    def _load(self):
        if not Path(self.filename).is_file():
            return {}

        with open(self.filename) as f:
            self.config = toml.load(f)

    def _validate(self):
        self.config = self.schema.validate(self.config)

    def save(self):
        self._validate()
        if not SAVE_CONFIG:
            return
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, "w") as f:
            toml.dump(self.config, f)


class VesslConfigLoader(ConfigLoader):
    schema = Schema(
        {
            Optional("user"): And(
                {
                    Optional("access_token"): str,
                    Optional("default_organization"): str,
                    Optional("default_project"): str,
                    Optional("workspace"): int,
                    Optional("cluster_kubeconfig"): str,
                    Optional("kubernetes_namespace"): str,
                    Optional("use_k0s_in_docker"): bool,
                },
                ignore_extra_keys=True,
            ),
        },
    )

    def __init__(self, path: str = DEFAULT_CONFIG_PATH):
        super().__init__(path, self.schema)

    def reset(self):
        self.config = {}
        self.save()

    @property
    def access_token(self):
        return self.config.get("user", {}).get("access_token")

    @access_token.setter
    def access_token(self, access_token: str):
        user = self.config.get("user", {})
        if access_token:
            user["access_token"] = access_token
        self.config["user"] = user
        self.save()

    @property
    def default_organization(self):
        return self.config.get("user", {}).get("default_organization")

    @default_organization.setter
    def default_organization(self, default_organization_name: str):
        user = self.config.get("user", {})
        if default_organization_name:
            user["default_organization"] = default_organization_name
        self.config["user"] = user
        self.save()

    @property
    def default_project(self):
        return self.config.get("user", {}).get("default_project")

    @default_project.setter
    def default_project(self, default_project_name: str):
        user = self.config.get("user", {})
        if default_project_name:
            user["default_project"] = default_project_name
        self.config["user"] = user
        self.save()

    @property
    def workspace(self):
        return self.config.get("user", {}).get("workspace")

    @workspace.setter
    def workspace(self, workspace_slug):
        user = self.config.get("user", {})
        if workspace_slug:
            user["workspace"] = workspace_slug
        self.config["user"] = user
        self.save()

    @property
    def cluster_kubeconfig(self) -> str:
        return self.config.get("user", {}).get("cluster_kubeconfig", "")

    @cluster_kubeconfig.setter
    def cluster_kubeconfig(self, cluster_kubeconfig_path: str):
        user = self.config.get("user", {})
        if cluster_kubeconfig_path:
            user["cluster_kubeconfig"] = cluster_kubeconfig_path
        self.config["user"] = user
        self.save()

    @property
    def kubernetes_namespace(self) -> str:
        return self.config.get("user", {}).get("kubernetes_namespace", "")

    @kubernetes_namespace.setter
    def kubernetes_namespace(self, kubernetes_namespace: str):
        user = self.config.get("user", {})
        if kubernetes_namespace:
            user["kubernetes_namespace"] = kubernetes_namespace
        self.config["user"] = user
        self.save()

    @property
    def use_k0s_in_docker(self) -> bool:
        return self.config.get("user", {}).get("use_k0s_in_docker", False)

    @use_k0s_in_docker.setter
    def use_k0s_in_docker(self, use_k0s_in_docker: bool):
        user = self.config.get("user", {})
        user["use_k0s_in_docker"] = use_k0s_in_docker
        self.config["user"] = user
        self.save()
