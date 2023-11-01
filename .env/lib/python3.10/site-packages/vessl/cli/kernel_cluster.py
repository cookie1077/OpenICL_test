import json
import os
import socket
from platform import uname
from shutil import which
from typing import List, Tuple, Optional

import click
import docker
from kubernetes import client as k8s_client

from openapi_client.models import ResponseKernelClusterInfo
from vessl.cli._base import VesslGroup, vessl_argument, vessl_option
from vessl.cli._util import (
    Endpoint,
    format_string,
    generic_prompter,
    print_data,
    print_debug,
    print_error,
    print_error_result,
    print_info,
    print_success,
    print_success_result,
    print_table,
    print_warning,
    prompt_choices,
    prompt_confirm,
)
from vessl.cli.organization import organization_name_option
from vessl.kernel_cluster import (
    CreateClusterParam,
    _acquire_kubeconfig,
    _check_existing_vessl_installation,
    _get_running_k0s_container,
    _load_cluster_kubeconfig,
    _load_kubeconfig_to_config,
    _start_k0s_container,
    _wait_for_k0s_ready,
    add_and_update_helm_repo,
    create_cluster,
    delete_cluster,
    list_cluster_nodes,
    list_clusters,
    read_cluster,
    rename_cluster,
)
from vessl.organization import _get_organization_name
from vessl.util.config import VesslConfigLoader
from vessl.util.constant import VESSL_HELM_REPO
from vessl.util.exception import (
    ACCESS_TOKEN_NOT_FOUND_ERROR_CODE,
    InvalidKernelClusterError,
    VesslApiException,
)


def cluster_name_prompter(ctx: click.Context, param: click.Parameter, value: str) -> str:
    clusters = list_clusters()
    return prompt_choices("Cluster", [x.name for x in clusters])


def custom_cluster_name_prompter(ctx: click.Context, param: click.Parameter, value: str) -> str:
    clusters = list_clusters()
    clusters = [x for x in clusters if not x.is_savvihub_managed]
    return prompt_choices("Cluster", [x.name for x in clusters])


@click.command(name="cluster", cls=VesslGroup)
def cli():
    pass


@cli.vessl_command()
@vessl_option(
    "--name",
    type=click.STRING,
    required=True,
    help="Name of the cluster. (e.g. 'seoul-cluster')",
    prompt="Enter cluster name",
)
@vessl_option(
    "--kubernetes-namespace",
    type=click.STRING,
    default="vessl",
    help="Kubernetes namespace to install VESSL agent (default: vessl)",
    prompt="Enter Kubernetes namespace to install VESSL agent",
    prompt_required=True,
)
@vessl_option(
    "--provider",
    type=click.Choice(["aws", "gcp", "on-premise"], case_sensitive=False),
    default="on-premise",
    help="Cluster provider; aws|gcp|on-premise (default: on-premise)",
    prompt="Select your cluster provider",
    prompt_required=True,
)
@vessl_option(
    "--extra-helm-values",
    type=click.STRING,
    multiple=True,
    default=[],
    help="Extra Helm values to pass when installing VESSL agent.",
)
@organization_name_option
def create(
    name: str,
    kubernetes_namespace: str,
    provider: str,
    extra_helm_values: List[str],
    organization_name: str = None,
):
    ########################################
    # validate inputs
    ########################################
    if name is None or len(name) < 4:
        print_error_result("Cluster name should be at least 4 characters long")
        return

    ########################################
    # validate pre-creation conditions
    ########################################
    if _check_helm_availability() == False:
        return
    print_info("Cluster installation dependencies are met.")

    config_path, context = _try_load_kubeconfig()
    if config_path is None:
        return
    print_info("Kubernetes configuration is loaded")
    print_data({"Kubeconfig path": config_path, "Current kubernetes context": context})

    if _vessl_already_exist(kubernetes_namespace):
        if not prompt_confirm(
            f"VESSL agent is already installed on current Kubernetes cluster. Do you want to remove and re-install?",
            default=False,
        ):
            return
        else:
            print_info("Cluster already has VESSL agent installed but force continuing...")
            print_warning("Removing existing VESSL agent...")
            remove_success = _try_remove_vessl_agent(kubernetes_namespace)
            if not remove_success:
                return

    else:
        print_info("VESSL cluster not found in your environment. Installing VESSL cluster...")

    ########################################
    # prepare helm repo
    ########################################
    if not prompt_confirm(
        f"Adding VESSL helm repo: {VESSL_HELM_REPO}. Continue?",
        default=True,
    ):
        print_error_result("Cluster creation aborted")
        print_info("You can add VESSL helm repo manually by running:")
        print_info(f"helm repo add vessl {VESSL_HELM_REPO}")
        return
    add_and_update_helm_repo()

    input_data = {
        "VESSL cluster name": name,
        "Provider": provider,
        "Kubeconfig path": config_path,
        "Current kubernetes context": context,
        "Kubernetes namespace": kubernetes_namespace,
    }

    ########################################
    # build create cluster param
    ########################################
    create_cluster_param = _build_create_cluster_param(
        name,
        kubernetes_namespace,
        provider,
        extra_helm_values,
        config_path,
        organization_name,
        input_data,
    )

    ########################################
    # user confirmation
    ########################################
    print_info("Cluster creation options:")
    print_data(input_data)
    if not prompt_confirm(
        f"Are you sure you want to install VESSL agent on current Kubernetes cluster with options displayed above?",
        default=True,
    ):
        print_error_result("Cluster creation aborted")
        return

    print_info("Creating cluster...")
    vessl_installation_info = create_cluster(create_cluster_param)
    VesslConfigLoader().kubernetes_namespace = kubernetes_namespace
    print_success_result(f"cluster {name} created!")
    print_cluster_data(vessl_installation_info)


def _build_create_cluster_param(
    name,
    kubernetes_namespace,
    provider,
    extra_helm_values,
    config_path,
    organization_name,
    input_data,
) -> CreateClusterParam:
    """Builds CreateClusterParam object from user inputs.\n
    Conditionally adds storage class, local path provisioner, and pod resource path.\n
    Also, sets input data for user confirmation.

    """
    create_cluster_param = CreateClusterParam()
    create_cluster_param.cluster_name = name
    create_cluster_param.kubernetes_namespace = kubernetes_namespace
    create_cluster_param.provider = provider
    create_cluster_param.kubeconfig_path = config_path
    if organization_name:
        create_cluster_param.organization_name = organization_name

    if provider == "gcp":
        print_debug(
            f"(provider=gcp) Setting default storage class for GCP as [standard-rwo]",
        )
        input_data["default storage class (gcp)"] = "standard-rwo"
    if provider == "aws":
        print_debug(
            f"(provider=aws) Setting default storage class for AWS as [vessl-ebs]",
        )
        input_data["default storage class (aws)"] = "vessl-ebs"

    # FIXME(mika): 추후 local path provisioner가 모든 클러스터에 마이그레이션 되면 아래의 값들을 전부 enable_longhorn으로 바꾼다.
    should_enable_local_path_provisioner = provider == "on-premise"
    print_debug(
        f"{'Enable' if should_enable_local_path_provisioner else 'Disable'} local-path-provisioner for {provider} cluster.\n"
        + "  This will allow VESSL agent to use host local storage for your workspace.\n"
    )
    input_data[f"local-path-provisioner"] = should_enable_local_path_provisioner
    if not should_enable_local_path_provisioner:
        # Default enabled for on-prem, disabled for cloud
        create_cluster_param.enable_local_path_provisioner = False
        # FIXME(mika): mitigation for non on-prem cluster installs.
        create_cluster_param.enable_longhorn = False

    if extra_helm_values and len(extra_helm_values) > 0:
        print_debug(
            f"Extra Helm values will be passed to VESSL agent installation: {extra_helm_values}",
        )
        input_data["Extra Helm values"] = ", ".join(extra_helm_values)
        create_cluster_param.extra_helm_values = extra_helm_values

    # dcgm patch
    v1 = k8s_client.CoreV1Api()
    # @XXX(seokju) rough assumption: k0s user will install k0s nodes
    # if this becomes a problem, patch chart and cli.
    node = v1.list_node().items[0]
    if "k0s" in node.status.node_info.kubelet_version:
        print_debug(
            f"(Required) Detected k0s node. Patching dcgm exporter podResourcePath as /var/lib/k0s/kubelete/pod-resources.",
        )
        input_data["(k0s) Patch dcgm exporter podResourcePath"] = True
        create_cluster_param.pod_resource_path = "/var/lib/k0s/kubelet/pod-resources"
    return create_cluster_param


def _vessl_already_exist(namespace: str):
    print_info("Checking existing VESSL installation...")
    try:
        vessl_installation_info = _check_existing_vessl_installation(namespace=namespace)
        if vessl_installation_info is not None:
            print_success("Existing VESSL installation found.")
            print_cluster_data(vessl_installation_info)
            return True
        return False
    except InvalidKernelClusterError as e:
        if e.code == ACCESS_TOKEN_NOT_FOUND_ERROR_CODE:
            print_error_result("Existing VESSL installation found, but access token is missing.")
        else:
            print_error_result(f"Existing VESSL installation found, but it is invalid. {repr(e)}")
        return True
    except VesslApiException as e:
        if e.status == 404:
            print_error(
                f"Existing VESSL installation, but not registered in VESSL Service: [{repr(e)}]"
            )
        else:
            print_error_result(
                f"Existing VESSL installation, but failed to check registration on VESSL Service: {repr(e)}"
            )
        return True


def _try_remove_vessl_agent(namespace: str) -> bool:
    config_path, context = _try_load_kubeconfig()
    if config_path is None:
        print_error_result("Failed to load kubeconfig")
        return False
    stream = os.popen(f"helm uninstall vessl --namespace {namespace} --kubeconfig {config_path}")
    output = stream.read()
    print_info(output)
    if stream.close() != None:
        print_error_result("Failed to uninstall vessl-agent")
        return False
    return True


def _try_load_kubeconfig():
    print_info("Loading cluster kubeconfig...")
    loaded, (config_path, context) = _load_cluster_kubeconfig()
    if not loaded:
        print_error_result("Kubernetes cluster not found;")
        print_error("Please configure valid kubeconfig file location by KUBECONFIG envvar.")
        print_info(
            "If you are a test user and want to install k0s-in-docker on your local machine, please run 'vessl cluster init-local' first."
        )
        return None, None

    return config_path, context


def _check_helm_availability():
    print_debug("Checking if helm exists...")
    if which("helm") is None:
        print_error_result(
            "Command 'helm' not found; Please install Helm to install VESSL cluster agent: https://helm.sh/"
        )
        return False
    print_debug("helm command available!")
    return True


@cli.vessl_command()
@vessl_argument("name", type=click.STRING, required=True, prompter=cluster_name_prompter)
@organization_name_option
def read(name: str):
    cluster = read_cluster(cluster_name=name)
    print_cluster_data(cluster)


@cli.vessl_command()
@organization_name_option
def list():
    clusters = list_clusters()
    print_table(
        clusters,
        ["ID", "Name", "Type", "Status", "K8s Master Endpoint", "K8s Namespace"],
        lambda x: [
            x.id,
            x.name,
            "Managed" if x.is_savvihub_managed else "Custom",
            x.status.replace("-", " "),
            format_string(x.name),
            format_string(x.kubernetes_namespace),
        ],
    )


@cli.vessl_command()
@vessl_argument("name", type=click.STRING, required=True, prompter=custom_cluster_name_prompter)
@organization_name_option
def delete(name: str):
    if not prompt_confirm(f"Are you sure you want to delete '{name}'?"):
        print_error_result("Cluster deletion aborted")
        return
    cluster = read_cluster(cluster_name=name)
    data = delete_cluster(cluster_id=cluster.id)
    print_success(f"Deleted '{name}'.")

    config = VesslConfigLoader()

    if prompt_confirm(f"Do you want to destroy the cluster installed with VESSL CLI?"):
        namespace = config.kubernetes_namespace
        if namespace is None:
            print_error_result(
                "Failed to find kubernetes namespace. Please check your kubeconfig file."
            )
        else:
            if not _try_remove_vessl_agent(namespace=namespace):
                print_error(
                    "Failed to remove vessl-agent. Please remove it manually by 'helm uninstall vessl --namespace <namespace>'"
                )
            else:
                print_success("vessl-agent removed successfully.")

    if config.use_k0s_in_docker:
        if _check_k0s_exists():
            print_info(
                "It seems that you are using k0s-in-docker installed by `vessl cluster init-local`."
            )
            if prompt_confirm(
                "Do you want to remove k0s-in-docker? (note that this is NOT RECOVERABLE)"
            ):
                is_error = _try_remove_existing_k0s()
                if is_error:
                    print_error_result("Failed to remove k0s-in-docker.")
                    print_info(
                        f"You can try execute the following in the shell (note that this is NOT RECOVERABLE)."
                    )
                    print_info(f"$ docker stop vessl-k0s && docker rm vessl-k0s")
                    print_info(
                        f'$ docker run -it --rm --privileged --pid=host alpine nsenter -t 1 -m -- sh -c "rm -rfv /var/lib/k0s"'
                    )
        else:
            config.use_k0s_in_docker = False


@cli.vessl_command()
@vessl_argument("name", type=click.STRING, required=True, prompter=custom_cluster_name_prompter)
@vessl_argument("new_name", type=click.STRING, required=True, prompter=generic_prompter("New name"))
@organization_name_option
def rename(name: str, new_name: str):
    cluster = read_cluster(cluster_name=name)
    cluster = rename_cluster(cluster_id=cluster.id, new_cluster_name=new_name)
    print_success(f"Renamed '{name}' to '{cluster.name}'.")


@cli.vessl_command()
@vessl_argument("name", type=click.STRING, required=True, prompter=custom_cluster_name_prompter)
@organization_name_option
def list_nodes(name: str):
    cluster = read_cluster(cluster_name=name)
    cluster_nodes = list_cluster_nodes(cluster_id=cluster.id)
    print_table(
        cluster_nodes,
        ["ID", "Name", "CPU Limits", "GPU Limits", "Memory Limits"],
        lambda x: [x.id, x.name, x.cpu_limits, x.gpu_limits, x.memory_limits],
    )


@cli.vessl_command()
def init_local():
    print_info("Checking if docker exists...")
    try:
        docker.from_env()
    except docker.errors.DockerException as e:
        print_error(
            f"[{repr(e)}] Docker not found; Please install Docker to run VESSL on your machine.: https://docs.docker.com/engine/install/"
        )
        return
    print_info("docker command available!")

    print_info("Checking if k0s-in-docker already exists...")
    if _check_k0s_exists():
        if prompt_confirm("k0s-in-docker already exists. Do you want to remove and reinstall it?"):
            is_error = _try_remove_existing_k0s()
            if is_error:
                print_error_result("Failed to remove k0s-in-docker.")
                return
            print_success("k0s-in-docker removed successfully.")
        else:
            print_info("You need to remove existing local cluster first.")
            print_info("Aborted.")
            return

    if not prompt_confirm(
        "Are you sure you want to initialize local cluster with k0s on docker?",
    ):
        print_info("Aborted.")
        return

    print_info(
        "Creating local cluster with k0s... (More Info: https://docs.k0sproject.io/v1.25.12+k0s.0/k0s-in-docker)"
    )
    kubeconfig, context_name, already_exists = _install_k0s_in_docker()
    if kubeconfig is None:
        print_error("Failed to create local cluster!")
        return
    if already_exists:
        print_info("Cluster already exists")
    else:
        print_info("Local cluster created!")

    VesslConfigLoader().use_k0s_in_docker = True
    print_success(
        "Initializing cluster success!, Hit `vessl cluster create` to rocket your Machine Learning!",
    )


def _check_k0s_exists():
    client = docker.from_env()
    containers = client.containers.list(filters={"name": "vessl-k0s"}, all=True)
    if len(containers) > 0:
        print_info("k0s-in-docker already exists!")
        print_data(
            {"Id": containers[0].id, "Name": containers[0].name, "Status": containers[0].status}
        )
        return True
    print_info("k0s-in-docker does not exist!")
    return False


def _try_remove_existing_k0s():
    client = docker.from_env()
    containers = client.containers.list(filters={"name": "vessl-k0s"}, all=True)
    is_error = False
    print_info("Removing k0s-in-docker...")
    try:
        for container in containers:
            if container.status == "running":
                container.stop()
            container.remove(force=True)
    except docker.errors.APIError as e:
        print_error(f"Failed to uninstall k0s-in-docker!: {repr(e)}")
        is_error = True
        return is_error
    print_info("Removing connected volume...")
    try:
        stream = os.popen(
            'docker run -it --rm --privileged --pid=host alpine nsenter -t 1 -m -- sh -c "rm -rfv /var/lib/k0s"'
        )
        output = stream.read()
        if stream.close() is not None:
            raise Exception(output)
        print_info(output)
    except Exception as e:
        print_error(f"Failed to remove connected volume!: {repr(e)}")
        print_info("You can try run following command manually;")
        print_info(
            '> docker run -it --rm --privileged --pid=host alpine nsenter -t 1 -m -- sh -c "rm -rfv /var/lib/k0s"'
        )
        is_error = True
        return is_error
    print_info("Uninstalled k0s-in-docker!")

    print_info("Removing custom kubeconfig...")
    try:
        kubeconfig_path = VesslConfigLoader().cluster_kubeconfig
        if os.path.exists(kubeconfig_path):
            os.remove(kubeconfig_path)
    except Exception as e:
        print_error(f"Failed to remove custom kubeconfig!: {repr(e)}")
        is_error = True
        return is_error
    print_info("Removed custom kubeconfig!")

    return is_error


def _install_k0s_in_docker() -> Tuple[Optional[str], Optional[str], bool]:
    """Install k0s-in-Docker on local machine.
    More Info: https://docs.k0sproject.io/v1.25.12+k0s.0/k0s-in-docker/
    """
    print_info(f"Installing k0s-in-docker...")
    k0s_config = """apiVersion: k0s.k0sproject.io/v1beta1
kind: ClusterConfig
metadata:
  name: k0s
spec:
  api:
    extraArgs:
      service-node-port-range: 32639-32767
"""
    kwargs = {
        "name": "vessl-k0s",
        "environment": {"K0S_CONFIG": k0s_config},
        "hostname": f"vessl-k0s-{socket.gethostname()}",
        "ports": {
            "6443/tcp": 6443,
        },
        "privileged": True,
        "volumes": {
            "/var/lib/k0s": {
                "bind": "/var/lib/k0s",
                "mode": "rw",
            },
        },
    }
    for i in range(32639, 32768):
        kwargs["ports"][f"{i}/tcp"] = i

    # Add extra arguments for arm64 machines
    if uname().machine == "arm64":
        print_info("System is using CPU with arm64 architecture; adding additional kwargs")
        kwargs["environment"]["ETCD_UNSUPPORTED_ARCH"] = "arm64"

    client = docker.from_env()

    # Add extra arguments for docker container running on Docker Desktop
    is_docker_desktop = client.info().get("OperatingSystem", "") == "Docker Desktop"
    if is_docker_desktop:
        print_info("System is running on Docker Desktop")

    if prompt_confirm(
        f"Should mount cgroup from host? (default: {'Y' if is_docker_desktop else 'N'}, Recommended for Docker Desktop)",
        default=is_docker_desktop,
    ):
        kwargs["cgroupns"] = "host"
        kwargs["volumes"]["/sys/fs/cgroup"] = {
            "bind": "/sys/fs/cgroup",
            "mode": "rw",
        }

    already_exists = False

    # Run k0s container
    print_info("k0s container configuration: ")
    print_warning(json.dumps(kwargs, indent=2))
    if not prompt_confirm("Starting k0s in docker with upper configuraion; Continue?"):
        print_info("Aborted.")
        return None, None, already_exists
    container = None
    try:
        container = _start_k0s_container(kwargs)
    except docker.errors.APIError as e:
        if e.status_code == 409 and "vessl-k0s" in e.explanation:
            print_error(
                "k0s-in-docker container already exists.",
            )
            already_exists = True
        else:
            print_error_result(
                f"[{repr(e)}] Failed to start k0s-in-docker container. Please check your docker configuration.",
            )
            return None, None, already_exists
    if already_exists:
        try:
            container = _get_running_k0s_container()
        except:
            print_error_result(
                "Failed to get running k0s-in-docker container. Please remove the existing container and try again.",
            )
            click.Abort()
            return None, None, already_exists

    # Wait for the container to be ready
    print_info(f"Waiting for the k0s-in-docker node to be ready...")
    try:
        _wait_for_k0s_ready(container)
    except Exception as e:
        click.secho(
            f"[{repr(e)}]; Error while waiting. Failed to initialize k0s-in-docker node.",
            bg="red",
            err=True,
        )
        click.Abort()
        return None, None, already_exists
    print_info(f"k0s-in-docker node is ready!")

    # Get kubeconfig, keep at /opt/vessl/cluster_kubeconfig.yaml for later
    print_info(f"Acquiring kubeconfig...")
    kubeconfig_filename = _acquire_kubeconfig(container)
    current_context = _load_kubeconfig_to_config(kubeconfig_filename)
    print_info(
        f"k0s-in-docker is ready! Saved kubeconfig at {kubeconfig_filename} for later use.",
    )
    return kubeconfig_filename, current_context["name"], already_exists


def cluster_id_prompter(ctx: click.Context, param: click.Parameter, value: str) -> int:
    clusters = list_clusters()
    cluster = prompt_choices(
        "Cluster",
        [(f"{x.name}", x) for x in clusters],
    )
    ctx.obj["cluster"] = cluster
    return cluster.id


def cluster_name_prompter(ctx: click.Context, param: click.Parameter, value: str) -> str:
    clusters = list_clusters()
    cluster = prompt_choices(
        "Cluster",
        [(f"{x.name}", x) for x in clusters],
    )
    ctx.obj["cluster"] = cluster
    return cluster.name


def cluster_name_callback(ctx: click.Context, param: click.Parameter, value: str) -> str:
    if "cluster" not in ctx.obj:
        ctx.obj["cluster"] = read_cluster(value)
    return value


def print_cluster_data(cluster: ResponseKernelClusterInfo, **kwargs):
    print_data(
        {
            "Name": cluster.name,
            "Dashboard URL": Endpoint.cluster.format(_get_organization_name(**kwargs), cluster.id),
            "Type": "VESSL-managed" if cluster.is_savvihub_managed else "Custom",
            "Region": format_string(cluster.region),
            "Status": cluster.status.replace("-", " "),
            "K8s Master Endpoint": format_string(cluster.name),
            "K8s Namespace": format_string(cluster.kubernetes_namespace),
            "K8s Service Type": cluster.kubernetes_service_type,
        }
    )


cluster_option = vessl_option(
    "-c",
    "--cluster",
    type=click.STRING,
    required=True,
    prompter=cluster_name_prompter,
    callback=cluster_name_callback,
    help="Must be specified before resource-related options (`--resource`, `--processor`, ...).",
)
