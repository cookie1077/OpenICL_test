import os
import tempfile
from pathlib import Path

import humanfriendly
from dotenv import dotenv_values

envvar = {**os.environ, **dotenv_values(".env")}

VESSL_ENV = envvar.get(
    "VESSL_ENV", "prod"
)  # default prod because cli and sdk should be packaged without .env file


# should be called at the end of this file as other values could be set across thie file, see below
# you should add any new envvar related constants here
def _print_envvar_related_constants_on_dev():
    if VESSL_ENV != "prod":
        print("Running in dev mode.. showing envvar related constants")
        print(
            f"""
>> VESSL_ENV: {VESSL_ENV}
>> WEB_HOST: {WEB_HOST}
>> API_HOST: {API_HOST}
>> SENTRY_DSN: {SENTRY_DSN}
>> DEFAULT_VESSL_SIDECAR_HOST: {DEFAULT_VESSL_SIDECAR_HOST}
>> PARALLEL_WORKERS: {PARALLEL_WORKERS}
"""
        )


VESSL_LOG_LEVEL_DEBUG = "DEBUG"
VESSL_LOG_LEVEL_INFO = "INFO"
VESSL_LOG_LEVEL_WARNING = "WARNING"
VESSL_LOG_LEVEL_ERROR = "ERROR"
VESSL_LOG_LEVEL_LEVELS = [
    VESSL_LOG_LEVEL_DEBUG,
    VESSL_LOG_LEVEL_INFO,
    VESSL_LOG_LEVEL_WARNING,
    VESSL_LOG_LEVEL_ERROR,
]

VESSL_LOG_LEVEL = (
    envvar.get("VESSL_LOG")
    if envvar.get("VESSL_LOG") in VESSL_LOG_LEVEL_LEVELS
    else VESSL_LOG_LEVEL_WARNING
)
DEFAULT_WEB_HOST = "https://vessl.ai" if VESSL_ENV == "prod" else "https://dev.vssl.ai"
WEB_HOST = envvar.get("VESSL_WEB_HOST", DEFAULT_WEB_HOST)
DEFAULT_API_HOST = "https://api.vessl.ai" if VESSL_ENV == "prod" else "https://api.dev.vssl.ai"
API_HOST = envvar.get("VESSL_API_HOST", DEFAULT_API_HOST)

SENTRY_DSN = envvar.get(
    "SENTRY_DSN",
    "https://e46fcd750b3a443fbd5b9dbc970e4ecf@o386227.ingest.sentry.io/5911639",
)
ENABLE_SENTRY = envvar.get("ENABLE_SENTRY", "true") == "true"

VESSL_HELM_REPO = "https://vessl-ai.github.io/helm-charts"
VESSL_HELM_CHART_NAME = "vessl"

K0S_VERSION = "v1.25.12-k0s.0"

DEFAULT_VESSL_SIDECAR_HOST = envvar.get("VESSL_SIDECAR_HOST", "http://localhost:3000")
UPDATE_CONTEXT_VARIABLE_URL = f"{DEFAULT_VESSL_SIDECAR_HOST}/store"
GET_ARGUMENT_VALUE_URL = f"{DEFAULT_VESSL_SIDECAR_HOST}/argument"
GET_CONTEXT_VARIABLE_URL = f"{DEFAULT_VESSL_SIDECAR_HOST}/context"

LOGIN_TIMEOUT_SECONDS = 160

VESSL_JWT_DIR = "/opt/vessl/access_token"
JWT_PATH = "/opt/vessl/access_token/token_file"
ACCESS_TOKEN_ENV_VAR = "VESSL_ACCESS_TOKEN"
DEFAULT_ORGANIZATION_ENV_VAR = "VESSL_DEFAULT_ORGANIZATION"
DEFAULT_PROJECT_ENV_VAR = "VESSL_DEFAULT_PROJECT"
CREDENTIALS_FILE_ENV_VAR = "VESSL_CREDENTIALS_FILE"
PROJECT_NAME_ENV_VAR = "VESSL_PROJECT_NAME"

CLUSTER_KUBECONFIG_ENV_VAR = "VESSL_CLUSTER_KUBECONFIG"
CLUSTER_MODE_SINGLE = "single"
CLUSTER_MODE_MULTI_NODE = "multi"

PROJECT_TYPE_VERSION_CONTROL = "version-control"
PROJECT_TYPES = [PROJECT_TYPE_VERSION_CONTROL]

DATASET_PATH_SCHEME_S3 = "s3://"
DATASET_PATH_SCHEME_GS = "gs://"

DATASET_VERSION_HASH_LATEST = "latest"

PROCESSOR_TYPE_CPU = "CPU"
PROCESSOR_TYPE_GPU = "GPU"
PROCESSOR_TYPES = [PROCESSOR_TYPE_CPU, PROCESSOR_TYPE_GPU]

SWEEP_OBJECTIVE_TYPE_MAXIMIZE = "maximize"
SWEEP_OBJECTIVE_TYPE_MINIMIZE = "minimize"
SWEEP_OBJECTIVE_TYPES = [SWEEP_OBJECTIVE_TYPE_MAXIMIZE, SWEEP_OBJECTIVE_TYPE_MINIMIZE]

SWEEP_ALGORITHM_TYPE_GRID = "grid"
SWEEP_ALGORITHM_TYPE_RANDOM = "random"
SWEEP_ALGORITHM_TYPE_BAYESIAN = "bayesian"
SWEEP_ALGORITHM_TYPES = [
    SWEEP_ALGORITHM_TYPE_GRID,
    SWEEP_ALGORITHM_TYPE_RANDOM,
    SWEEP_ALGORITHM_TYPE_BAYESIAN,
]

SWEEP_PARAMETER_TYPE_CATEGORICAL = "categorical"
SWEEP_PARAMETER_TYPE_INT = "int"
SWEEP_PARAMETER_TYPE_DOUBLE = "double"
SWEEP_PARAMETER_TYPES = [
    SWEEP_PARAMETER_TYPE_CATEGORICAL,
    SWEEP_PARAMETER_TYPE_INT,
    SWEEP_PARAMETER_TYPE_DOUBLE,
]

SWEEP_PARAMETER_RANGE_TYPE_SPACE = "space"
SWEEP_PARAMETER_RANGE_TYPE_LIST = "list"
SWEEP_PARAMETER_RANGE_TYPES = [
    SWEEP_PARAMETER_RANGE_TYPE_SPACE,
    SWEEP_PARAMETER_RANGE_TYPE_LIST,
]

MODEL_SOURCE_EXPERIMENT = "experiment"
MODEL_SOURCE_LOCAL = "local"

SOURCE_TYPE_CODE = "code"
SOURCE_TYPE_ARCHIVE_FILE = "archive-file"
SOURCE_TYPE_OBJECT_STORAGE = "object-storage"
SOURCE_TYPE_DATASET = "dataset"
SOURCE_TYPE_DATASET_VERSION = "dataset-version"
SOURCE_TYPE_MODEL_VOLUME = "model-volume"
SOURCE_TYPE_EMPTY_DIR = "empty-dir"
SOURCE_TYPE_OUTPUT = "output"
SOURCE_TYPE_PROJECT = "project"
SOURCE_TYPE_OUTPUT = "output"
SOURCE_TYPE_OBJECT_STORAGE = "object-storage"

MOUNT_PATH_EMPTY_DIR = "/root/"
MOUNT_PATH_OUTPUT = "/output/"
MOUNT_PATH_PROJECT = "/root/{}"

EXPERIMENT_WORKING_DIR = "/root/"

FRAMEWORK_TYPE_PYTORCH = "pytorch"
FRAMEWORK_TYPE_TENSORFLOW = "tensorflow"
FRAMEWORK_TYPES = (FRAMEWORK_TYPE_PYTORCH, FRAMEWORK_TYPE_TENSORFLOW)

PARALLEL_WORKERS = envvar.get("VESSL_PARALLEL_WORKERS", 20)

VESSL_MEDIA_PATH = "vessl-media"
VESSL_IMAGE_PATH = "images"
VESSL_AUDIO_PATH = "audio"
VESSL_PLOTS_FILETYPE_IMAGE = "image"
VESSL_PLOTS_FILETYPE_IMAGES = "images"
VESSL_PLOTS_FILETYPE_AUDIO = "audio"

WORKSPACE_BACKUP_MAX_SIZE = 15 * 1024 * 1024 * 1024
WORKSPACE_BACKUP_MAX_SIZE_FORMATTED = humanfriendly.format_size(
    WORKSPACE_BACKUP_MAX_SIZE, binary=True
)

TEMP_DIR = tempfile.gettempdir()

SSH_CONFIG_PATH = os.path.join(Path.home(), ".ssh", "config")
SSH_CONFIG_FORMAT = """Host {host}
    User root
    Hostname {hostname}
    Port {port}
    StrictHostKeyChecking accept-new
    CheckHostIP no
"""
SSH_PUBLIC_KEY_PATH = os.path.join(Path.home(), ".ssh", "id_ed25519.pub")
SSH_PRIVATE_KEY_PATH = os.path.join(Path.home(), ".ssh", "id_ed25519")


class colors:
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    GREY = "\033[38;5;244m"


LOGO = f"""{colors.OKCYAN}
        $$\    $$\ $$$$$$$$\  $$$$$$\   $$$$$$\  $$\              $$$$$$\  $$$$$$\ 
        $$ |   $$ |$$  _____|$$  __$$\ $$  __$$\ $$ |            $$  __$$\ \_$$  _|
        $$ |   $$ |$$ |      $$ /  \__|$$ /  \__|$$ |            $$ /  $$ |  $$ |  
        \$$\  $$  |$$$$$\    \$$$$$$\  \$$$$$$\  $$ |            $$$$$$$$ |  $$ |  
         \$$\$$  / $$  __|    \____$$\  \____$$\ $$ |            $$  __$$ |  $$ |  
          \$$$  /  $$ |      $$\   $$ |$$\   $$ |$$ |            $$ |  $$ |  $$ |  
           \$  /   $$$$$$$$\ \$$$$$$  |\$$$$$$  |$$$$$$$$\       $$ |  $$ |$$$$$$\ 
            \_/    \________| \______/  \______/ \________|      \__|  \__|\______|
        {colors.OKCYAN}"""

NANO_GPT = """
name: nanogpt
image: nvcr.io/nvidia/pytorch:22.03-py3
resources:
  cluster: aws-apne2
  preset: v1.a10g-1.mem-26
import:
  /root/examples: git://github.com/vessl-ai/examples
export:
  /output: vessl-artifact://
run:
  - workdir: /root/examples/nanogpt
    command: |
      pip install torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
      pip install transformers datasets tiktoken wandb tqdm
      python data/shakespeare_char/prepare.py
      python train.py config/train_shakespeare_char.py
      python sample.py --out_dir=out-shakespeare-char
"""
STABLE_DIFFUSION = """
name : stable-diffusion
resources:
  cluster: aws-apne2
  preset: v1.a10g-1.mem-26
image: quay.io/vessl-ai/ngc-pytorch-kernel:22.12-py3-202301160809
run:
  - workdir: /root/examples/stable-diffusion
    command: |
      bash ./setup.sh && streamlit run main.py
import:
  /root/examples: git://github.com/vessl-ai/examples
interactive:
  max_runtime: 24h
  jupyter:
    idle_timeout: 120m
ports:
  - 8501
"""

SEGMENT_ANYTHING = """
name : segment-anything
resources:
  cluster: aws-apne2
  preset: v1.a10g-1.mem-26
image: quay.io/vessl-ai/ngc-pytorch-kernel:22.12-py3-202301160809
run:
  - workdir: /root/examples/segment-anything
    command: |
      bash ./setup.sh
import:
  /root/examples: git://github.com/vessl-ai/examples
interactive:
  max_runtime: 24h
  jupyter:
    idle_timeout: 120m
ports:
  - 8501
"""

LANGCHAIN = """
name : langchain
resources:
  cluster: aws-apne2
  preset: v1.cpu-4.mem-13
image: quay.io/vessl-ai/kernels:py39-202309261114
run:
  - workdir: /root/examples/langchain/question_answering
    command: |
      bash ./run.sh
import:
  /root/examples: git://github.com/vessl-ai/examples
interactive:
  max_runtime: 24h
  jupyter:
    idle_timeout: 120m
ports:
  - 8501
"""
THIN_PLATE_SPLINE_MOTION = """
name: Thin-Plate-Spline-Motion-Model
image: nvcr.io/nvidia/pytorch:21.05-py3
resources:
  cluster: aws-apne2
  preset: v1.v100-1.mem-52
run:
  - workdir: /root/examples/thin-plate-spline-motion-model
    command: |
      pip install -r requirements.txt
      pip install vessl
      python run.py --config config/vox-256.yaml --device_ids 0
      cp -r ./logs /output
import:
  /root/examples: git://github.com/vessl-ai/examples
  /root/examples/vox: s3://vessl-public-apne2/vessl_run_datasets/vox/
export:
  /output: vessl-artifact://
"""
LLAMA_2_PLAYGROUND = """
name: llama2_c_playground
description: Batch inference with llama2.c.
tags:
  - playground
  - inference
resources:
  cluster: aws-apne2
  preset: v1.cpu-2.mem-6
image: quay.io/vessl-ai/ngc-pytorch-kernel:23.07-py3-202308010607
import:
  /root/examples/: git://github.com/vessl-ai/examples.git
run:
  - command: pip install streamlit
    workdir: /root/examples
  - command: wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
    workdir: /root/examples/llama2_c
  - command: streamlit run llama2_c/streamlit/llama2_c_inference.py --server.port=80
    workdir: /root/examples
interactive:
  max_runtime: 24h
  jupyter:
    idle_timeout: 120m
ports:
  - name: streamlit
    type: http
    port: 80
"""
LLAMA_2_TRAIN = """
name: llama2_c_training
description: Training llama2_c from scratch with VESSL Run.
tags:
  - training
resources:
  cluster: aws-apne2
  preset: v1.a10g-1.mem-26
image: quay.io/vessl-ai/ngc-pytorch-kernel:23.07-py3-202308010607
import:
  /root/llama2_c/: git://github.com/karpathy/llama2.c.git
export:
  /output/: vessl-artifact://
run:
  - command: |-
      pip install -r requirements.txt && \
      python tinystories.py download && \
      python tinystories.py pretokenize && \
      python train.py --dtype=float16 --compile=False --eval_iters=$eval_iters --batch_size=$batch_size --max_iters=$max_iters --out_dir=$out_dir
    workdir: /root/llama2_c
ports: []
env:
  batch_size: "8"
  dtype: float16
  eval_iters: "10"
  max_iters: "10000"
  out_dir: /output
service_account_name: ""
termination_protection: false
"""
LLAMA_2_FINE_TUNING = """
name: train_tolkien_llama_2
resources:
  cluster: aws-apne2
  preset: v1.v100-1.mem-52
image: quay.io/vessl-ai/ngc-pytorch-kernel:23.09-py3-202310150329
import:
  /ckpt/: vessl-model://vessl-ai/llama2/1
  /root/examples/: git://github.com/vessl-ai/examples.git
run:
  - command: |-
      pip install --upgrade pip
      pip install transformers datasets accelerate peft bitsandbytes sentencepiece -qq
      wget https://raw.githubusercontent.com/jeremyarancio/llm-tolkien/main/llm/data/extracted_text.jsonl
      cd /ckpt/
      unzip llama_2_7b_hf.zip
      cd /root/examples/LLM_llama_2_LOTR
      python train_tolkien_llama_2.py
    workdir: /root/examples/LLM_llama_2_LOTR
"""

_print_envvar_related_constants_on_dev()
