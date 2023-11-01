import os
import time
from typing import List, Optional

import click

from vessl.cli._base import VesslGroup, vessl_argument, vessl_option
from vessl.cli._util import (
    Endpoint,
    generic_prompter,
    print_data,
    print_info,
    print_logs,
    print_success,
    print_table,
    print_volume_files,
    prompt_choices,
    truncate_datetime,
)
from vessl.cli.kernel_cluster import cluster_option
from vessl.cli.kernel_resource_spec import resource_option
from vessl.cli.organization import organization_name_option
from vessl.cli.project import project_name_option
from vessl.experiment import (
    create_experiment,
    delete_experiment,
    download_experiment_output_files,
    list_experiment_logs,
    list_experiment_output_files,
    list_experiments,
    read_experiment,
    terminate_experiment,
)
from vessl.kernel_cluster import list_cluster_nodes
from vessl.kernel_image import list_kernel_images
from vessl.organization import list_organization_credentials
from vessl.util import logger
from vessl.util.constant import (
    EXPERIMENT_WORKING_DIR,
    FRAMEWORK_TYPE_PYTORCH,
    FRAMEWORK_TYPE_TENSORFLOW,
    FRAMEWORK_TYPES,
    MOUNT_PATH_OUTPUT,
    PROCESSOR_TYPE_GPU,
    PROCESSOR_TYPES,
)


def experiment_number_prompter(
    ctx: click.Context,
    param: click.Parameter,
    value: str,
) -> str:
    experiments = list_experiments()
    return prompt_choices("Experiment", [(f"#{x.number}", x.number) for x in reversed(experiments)])


def local_experiment_number_prompter(
    ctx: click.Context,
    param: click.Parameter,
    value: str,
) -> str:
    experiments = [e for e in list_experiments() if e.is_local]
    return prompt_choices("Experiment", [(f"#{x.number}", x.number) for x in reversed(experiments)])


def processor_type_prompter(ctx: click.Context, param: click.Parameter, value: str) -> str:
    cluster = ctx.obj.get("cluster")
    if cluster is None:
        raise click.BadOptionUsage(
            option_name="--cluster",
            message="Cluster (`--cluster`) must be specified before processor type (`--processor-type`).",
        )

    if ctx.obj.get("resource") is None:
        processor_type = prompt_choices("Processor Type", PROCESSOR_TYPES)
        ctx.obj["processor_type"] = processor_type
        return processor_type


def processor_type_callback(ctx: click.Context, param: click.Parameter, value: str):
    if value:
        ctx.obj["processor_type"] = value
    return value


def cpu_limit_prompter(ctx: click.Context, param: click.Parameter, value: float) -> float:
    cluster = ctx.obj.get("cluster")
    if cluster is None:
        raise click.BadOptionUsage(
            option_name="--cluster",
            message="Cluster (`--cluster`) must be specified before CPU limit (`--cpu-limit`).",
        )

    if ctx.obj.get("resource") is None:
        return click.prompt("CPUs (in vCPU)", type=click.FLOAT)


def memory_limit_prompter(ctx: click.Context, param: click.Parameter, value: float) -> float:
    cluster = ctx.obj.get("cluster")
    if cluster is None:
        raise click.BadOptionUsage(
            option_name="--cluster",
            message="Cluster (`--cluster`) must be specified before memory limit (`--memory-limit`).",
        )

    if ctx.obj.get("resource") is None:
        return click.prompt("Memory (e.g. 4Gi)", type=click.STRING)


def gpu_type_prompter(ctx: click.Context, param: click.Parameter, value: str) -> str:
    cluster = ctx.obj.get("cluster")
    if cluster is None:
        raise click.BadOptionUsage(
            option_name="--cluster",
            message="Cluster (`--cluster`) must be specified before GPU type (`--gpu-type`).",
        )

    if ctx.obj.get("resource") is None:
        processor_type = ctx.obj.get("processor_type")
        if processor_type is None:
            raise click.UsageError(
                message="Processor type must be specified before GPU type (`--gpu-type`).",
            )
        if processor_type == PROCESSOR_TYPE_GPU:
            nodes = list_cluster_nodes(cluster.id)
            return prompt_choices("GPU Type", [x.gpu_product_name for x in nodes])


def gpu_limit_prompter(ctx: click.Context, param: click.Parameter, value: float) -> float:
    cluster = ctx.obj.get("cluster")
    if cluster is None:
        raise click.BadOptionUsage(
            option_name="--cluster",
            message="Cluster (`--cluster`) must be specified before GPU limit (`--gpu-limit`).",
        )

    if ctx.obj.get("resource") is None:
        processor_type = ctx.obj.get("processor_type")
        if processor_type is None:
            raise click.UsageError(
                message="Processor type must be specified before GPU limit (`--gpu-limit`).",
            )

        if processor_type == PROCESSOR_TYPE_GPU:
            return click.prompt("GPUs (in vGPU)", type=click.FLOAT)


def image_url_prompter(ctx: click.Context, param: click.Parameter, value: str) -> str:
    processor_type = ctx.obj.get("processor_type")
    if processor_type is None:
        raise click.UsageError(
            message="Processor type must be specified before image URL (`--image-url`).",
        )

    choice = prompt_choices(
        "Image type",
        [
            "VESSL Managed Docker Image",
            "Public Docker Image",
            "Private Docker Image",
        ],
    )

    if choice == "Public Docker Image":
        ctx.obj["is_image_private"] = False
        return click.prompt("Image URL", type=click.STRING)
    elif choice == "Private Docker Image":
        ctx.obj["is_image_private"] = True
        return click.prompt("Image URL", type=click.STRING)
    else:
        ctx.obj["is_image_private"] = False
        images = list_kernel_images()
        images = [x for x in images if x.processor_type == processor_type]

        # Pretty print image name+URL choices
        name_maxlen = len(max(images, key=lambda x: len(x.name)).name)
        return prompt_choices(
            "Image",
            [
                (
                    f"{x.name}" + " " * (name_maxlen - len(x.name) + 2) + f"(URL: {x.image_url})",
                    x.image_url,
                )
                for x in images
            ],
        )


def upload_local_file_callback(
    ctx: click.Context, param: click.Parameter, value: List[str]
) -> List[str]:
    if value:
        for dir_spec in value:
            if ":" in dir_spec:
                local_path, _ = dir_spec.split(":")
            else:
                local_path = dir_spec

            if not local_path:
                raise click.BadOptionUsage(
                    option_name="--upload-local-file",
                    message=f"Invalid path {local_path}.",
                )

            if not os.path.exists(local_path):
                raise click.BadOptionUsage(
                    option_name="--upload-local-file",
                    message=f"{local_path} does not exist.",
                )
    return value


def docker_credentials_prompter(
    ctx: click.Context, param: click.Parameter, value: int
) -> Optional[int]:
    if ctx.obj.get("is_image_private"):
        creds = list_organization_credentials(
            types=["aws-access-key,docker-credentials"],
            components=["image,image"],
        )
        return prompt_choices(
            "Image",
            [
                (
                    f"{c.credentials_name} | {c.credentials_type} | (created by: '{c.created_by.username}')",
                    c.credentials_id,
                )
                for c in creds
            ],
        )
    return None


def worker_count_callback(ctx: click.Context, param: click.Parameter, value: int) -> int:
    if value is None:
        value = 1

    if value < 1:
        raise click.BadOptionUsage(
            option_name="--worker-count",
            message="num nodes (`--num-nodes`) must be a positive integer.",
        )

    ctx.obj["worker_count"] = value
    return value


def framework_type_prompter(ctx: click.Context, param: click.Parameter, value: str) -> str:
    worker_count = ctx.obj.get("worker_count")
    if worker_count == 1:
        return ""

    framework_type = prompt_choices("Processor Type", FRAMEWORK_TYPES)
    if framework_type == FRAMEWORK_TYPE_TENSORFLOW:
        raise click.BadOptionUsage(
            option_name="--framework-type",
            message="Only PyTorch distributed experiment is supported currently.",
        )
    return framework_type


@click.command(name="experiment", cls=VesslGroup)
def cli():
    pass


@cli.vessl_command()
@vessl_argument(
    "number",
    type=click.INT,
    required=True,
    prompter=experiment_number_prompter,
)
@organization_name_option
@project_name_option
def read(number: int):
    experiment = read_experiment(experiment_number=number)

    distributed_spec = "None"
    if experiment.is_distributed:
        if experiment.distributed_spec.framework_type == FRAMEWORK_TYPE_PYTORCH:
            distributed_spec = {
                "Framework Type": experiment.distributed_spec.framework_type,
                "PyTorch Spec": {
                    "Worker count": experiment.distributed_spec.pytorch_spec.worker_replicas,
                },
            }
        elif experiment.distributed_spec.framework_type == FRAMEWORK_TYPE_TENSORFLOW:
            distributed_spec = {
                "Framework Type": experiment.distributed_spec.framework_type,
                "TensorFlow Spec": experiment.distributed_spec.tensorflow_spec,
            }

    kernel_image = "None"
    if experiment.kernel_image:
        kernel_image = {
            "Name": experiment.kernel_image.name,
            "URL": experiment.kernel_image.image_url,
        }

    resource_spec = "None"
    if experiment.kernel_resource_spec:
        resource_spec = {
            "Name": experiment.kernel_resource_spec.name,
            "CPU Type": experiment.kernel_resource_spec.cpu_type,
            "CPU Limit": experiment.kernel_resource_spec.cpu_limit,
            "Memory Limit": experiment.kernel_resource_spec.memory_limit,
            "GPU Type": experiment.kernel_resource_spec.gpu_type,
            "GPU Limit": experiment.kernel_resource_spec.gpu_limit,
        }

    metrics_summary = "None"
    if experiment.metrics_summary.latest:
        metrics_keys = experiment.metrics_summary.latest.keys()
        metrics_summary = {}
        for key in metrics_keys:
            metrics_summary[key] = experiment.metrics_summary.latest[key].value

    print_data(
        {
            "ID": experiment.id,
            "Number": experiment.number,
            "Distributed": experiment.is_distributed,
            "Distributed spec": distributed_spec,
            "Local": experiment.is_local,
            "Status": experiment.status,
            "Created": truncate_datetime(experiment.created_dt),
            "Message": experiment.message,
            "Kernel image": kernel_image,
            "Resource spec": resource_spec,
            "Start command": experiment.start_command,
            "Metrics summary": metrics_summary,
        }
    )
    print_info(
        f"For more info: {Endpoint.experiment.format(experiment.organization.name, experiment.project.name, experiment.number)}"
    )


@cli.vessl_command()
@organization_name_option
@project_name_option
def list():
    experiments = list_experiments()
    print_table(
        experiments,
        ["ID", "Number", "Distributed", "Status", "Created", "Message"],
        lambda x: [
            x.id,
            x.number,
            x.is_distributed,
            x.status,
            truncate_datetime(x.created_dt),
            x.message,
        ],
    )


command_option = vessl_option(
    "-x",
    "--command",
    type=click.STRING,
    required=True,
    prompter=generic_prompter("Start command"),
    help="Start command to execute in experiment container.",
)
processor_type_option = vessl_option(
    "--processor-type",
    type=click.Choice(("CPU", "GPU")),
    prompter=processor_type_prompter,
    callback=processor_type_callback,
    help="CPU or GPU (for custom resource only).",
)
cpu_limit_option = vessl_option(
    "--cpu-limit",
    type=click.FLOAT,
    prompter=cpu_limit_prompter,
    help="Number of vCPUs (for custom resource only).",
)
memory_limit_option = vessl_option(
    "--memory-limit",
    type=click.STRING,
    prompter=memory_limit_prompter,
    help="Memory limit (e.g. 4Gi) (for custom resource only).",
)
gpu_type_option = vessl_option(
    "--gpu-type",
    type=click.STRING,
    prompter=gpu_type_prompter,
    help="GPU type such as Tesla-K80 (for custom resource only).",
)
gpu_limit_option = vessl_option(
    "--gpu-limit",
    type=click.INT,
    prompter=gpu_limit_prompter,
    help="Number of GPU cores (for custom resource only).",
)
image_url_option = vessl_option(
    "-i",
    "--image-url",
    type=click.STRING,
    prompter=image_url_prompter,
    help="Kernel docker image URL",
)
docker_credentials_id_option = vessl_option(
    "--docker-credentials-id",
    type=click.INT,
    prompter=docker_credentials_prompter,
    help="Docker credential prompter",
)
message_option = click.option("-m", "--message", type=click.STRING)
termination_protection_option = click.option("--termination-protection", is_flag=True)
hyperparameter_option = click.option(
    "-h",
    "--hyperparameter",
    type=click.STRING,
    multiple=True,
    help="Hyperparameters. Format: [key]=[value], ex. `--hyperparameter lr=0.01`.",
)
secret_option = click.option(
    "--secret",
    type=click.STRING,
    multiple=True,
    help=(
        "Secret environment variables that are hidden in UI and logs. "
        "Format: [key]=[value], ex. `--secret DB_PASSWORD=vessl1234`."
    ),
)
dataset_option = click.option(
    "--dataset",
    type=click.STRING,
    multiple=True,
    help="Dataset mounts. Format: [mount_path]:[dataset_name]@[optional_dataset_version], ex. `--dataset /input:mnist@3bcd5f`.",
)
model_option = click.option(
    "--model",
    type=click.STRING,
    multiple=True,
    help="Model mounts. Format: [mount_path]:[model_repository_name]/[model_number], ex. `--model /input:pytorch-model/1` .",
)
git_ref_option = click.option(
    "--git-ref",
    type=click.STRING,
    multiple=True,
    help=f"Git repository mounts. Format: [mount_path]:github/[organization]/[repository]/[optional_commit], ex. `--git-ref {EXPERIMENT_WORKING_DIR}examples:github/vessl-ai/examples/3cd23dd`.",
)
git_diff_option = click.option(
    "--git-diff",
    type=click.STRING,
    help="Git diff file mounts. Format: [mount_path]:[volume_file_path]. This option is used only for reproducing existing experiments.",
)
upload_local_file_option = vessl_option(
    "--upload-local-file",
    type=click.STRING,
    multiple=True,
    callback=upload_local_file_callback,
    help="Upload local file. Format: [local_path] or [local_path]:[remote_path].",
)
use_vesslignore_option = vessl_option(
    "--use-vesslignore/--no-use-vesslignore",
    type=click.BOOL,
    default=True,
    help="\n".join(
        [
            "Choose whether to use .vesslignore files when uploading local files.",
            "A .vesslignore file contains patterns that applies relatively to the directory containing that .vesslignore file. Files that match any of those patterns are excluded from uploading.",
            "The syntax of .vesslignore is similar to that of .gitignore. Please consult the documentation for more details.",
        ]
    ),
)
upload_local_git_diff_option = vessl_option(
    "--upload-local-git-diff",
    is_flag=True,
    help="Upload local git commit hash and diff.",
)
archive_file_option = click.option(
    "--archive-file",
    type=click.STRING,
    help="Local archive file mounts. Format: [mount_path]:[archive_file_path]]. This option is used only for reproducing existing experiments.",
)
object_storage_option = vessl_option(
    "--object-storage",
    type=click.STRING,
    multiple=True,
    help="[ALPHA] Object storage mounts. Format: [upload|download]@[mount_path]:[object_storage_name]/[object_storage_path]. e.g. `--object-storage upload@/input:s3://some/bucket.path`.",
)
root_volume_size_option = click.option("--root-volume-size", type=click.STRING)
working_dir_option = click.option(
    "--working-dir", type=click.STRING, help=f"Defaults to `{EXPERIMENT_WORKING_DIR}`."
)
output_dir_option = click.option(
    "--output-dir",
    type=click.STRING,
    default=MOUNT_PATH_OUTPUT,
    help="Directory to store experiment output files. Defaults to `/output/`.",
)
worker_count_option = vessl_option(
    "--worker-count",
    type=click.INT,
    callback=worker_count_callback,
    help="The number of nodes to run an experiment. Defaults to 1.",
)
framework_type_option = vessl_option(
    "--framework-type",
    type=click.STRING,
    prompter=framework_type_prompter,
    help="Framework type option. Defaults to `pytorch`.",
)
service_account_name_option = vessl_option(
    "--service-account",
    type=click.STRING,
    default="",
    help="Kubernetes service account name.",
)


@cli.vessl_command()
@click.pass_context
@organization_name_option
@project_name_option
@cluster_option
@vessl_option(
    "--node",
    type=click.STRING,
    multiple=True,
    help="Cluster nodes. Defaults to all nodes in cluster.",
)
@resource_option
@processor_type_option
@cpu_limit_option
@memory_limit_option
@gpu_type_option
@gpu_limit_option
@image_url_option
@docker_credentials_id_option
@message_option
@termination_protection_option
@hyperparameter_option
@secret_option
@dataset_option
@model_option
@git_ref_option
@git_diff_option
@upload_local_file_option
@use_vesslignore_option
@upload_local_git_diff_option
@archive_file_option
@object_storage_option
@root_volume_size_option
@working_dir_option
@output_dir_option
@command_option
@worker_count_option
@framework_type_option
@service_account_name_option
def create(
    ctx,
    cluster: str,
    node: List[str],
    command: str,
    resource: str,
    processor_type: str,
    cpu_limit: float,
    memory_limit: str,
    gpu_type: str,
    gpu_limit: int,
    image_url: str,
    docker_credentials_id: Optional[int],
    message: str,
    termination_protection: bool,
    hyperparameter: List[str],
    secret: List[str],
    dataset: List[str],
    model: List[str],
    git_ref: List[str],
    git_diff: str,
    upload_local_file: List[str],
    use_vesslignore: bool,
    upload_local_git_diff: bool,
    archive_file: str,
    object_storage: List[str],
    root_volume_size: str,
    working_dir: str,
    output_dir: str,
    worker_count: int,
    framework_type: str,
    service_account: str,
):
    experiment = create_experiment(
        cluster_name=cluster,
        cluster_node_names=node,
        start_command=command,
        kernel_resource_spec_name=resource,
        processor_type=processor_type,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        gpu_type=gpu_type,
        gpu_limit=gpu_limit,
        kernel_image_url=image_url,
        docker_credentials_id=docker_credentials_id,
        message=message,
        termination_protection=termination_protection,
        hyperparameters=hyperparameter,
        secrets=secret,
        dataset_mounts=dataset,
        model_mounts=model,
        git_ref_mounts=git_ref,
        git_diff_mount=git_diff,
        local_files=upload_local_file,
        use_vesslignore=use_vesslignore,
        upload_local_git_diff=upload_local_git_diff,
        archive_file_mount=archive_file,
        object_storage_mounts=object_storage,
        root_volume_size=root_volume_size,
        working_dir=working_dir,
        output_dir=output_dir,
        worker_count=worker_count,
        framework_type=framework_type,
        service_account=service_account,
    )
    ctx.obj["experiment_number"] = experiment.number
    print_success(
        f"Created '{experiment.number}'.\n"
        f"For more info: {Endpoint.experiment.format(experiment.organization.name, experiment.project.name, experiment.number)}"
    )


worker_number_option = vessl_option(
    "--worker-number",
    type=click.INT,
    default=0,
    help="Worker number (for distributed experiment only).",
)


@cli.vessl_command()
@vessl_argument(
    "number",
    type=click.INT,
    required=True,
    prompter=experiment_number_prompter,
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
@worker_number_option
def logs(
    number: int,
    tail: int,
    follow: bool,
    worker_number: int,
):
    if not follow:
        logs = list_experiment_logs(experiment_number=number, tail=tail, worker_numer=worker_number)
        print_logs(logs)
        print_info(f"Displayed last {len(logs)} lines of #{number}.")
        return

    after = 0

    experiment_finished_dt = None
    while True:
        if (
            read_experiment(number).status not in ["pending", "running"]
            and experiment_finished_dt is None
        ):
            experiment_finished_dt = time.time()

        if experiment_finished_dt is not None and (time.time() - experiment_finished_dt) > 5:
            break

        logs = list_experiment_logs(
            experiment_number=number,
            before=int(time.time() - 5),
            after=after,
            worker_numer=worker_number,
        )
        print_logs(logs)
        if len(logs) > 0:
            after = logs[-1].timestamp + 0.000001

        time.sleep(3)


@cli.vessl_command()
@vessl_argument(
    "number",
    type=click.INT,
    required=True,
    prompter=experiment_number_prompter,
)
@organization_name_option
@project_name_option
@worker_number_option
def list_output(
    number: int,
    worker_number: int,
):
    files = list_experiment_output_files(
        experiment_number=number,
        need_download_url=False,
        recursive=True,
        worker_number=worker_number,
    )
    print_volume_files(files)


@cli.vessl_command()
@vessl_argument(
    "number",
    type=click.INT,
    required=True,
    prompter=experiment_number_prompter,
)
@click.option(
    "-p",
    "--path",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "output"),
    help="Path to store downloads. Defaults to `./output`.",
)
@worker_number_option
@organization_name_option
@project_name_option
def download_output(
    number: int,
    path: str,
    worker_number: int,
):
    download_experiment_output_files(
        experiment_number=number,
        dest_path=path,
        worker_number=worker_number,
    )
    print_success(f"Downloaded experiment output to {path}.")


@cli.vessl_command()
@vessl_argument(
    "number",
    type=click.INT,
    required=True,
    prompter=experiment_number_prompter,
)
@organization_name_option
@project_name_option
def terminate(number: int):
    experiment = terminate_experiment(experiment_number=number)
    print_success(
        f"Terminated '#{experiment.number}'.\n"
        f"For more info: {Endpoint.experiment.format(experiment.organization.name, experiment.project.name, experiment.number)}"
    )


@cli.vessl_command()
@vessl_argument(
    "number",
    type=click.INT,
    required=True,
    prompter=experiment_number_prompter,
)
@organization_name_option
@project_name_option
def delete(number: int):
    delete_experiment(experiment_number=number)
    print_success(f"Deleted '#{number}'.\n")
