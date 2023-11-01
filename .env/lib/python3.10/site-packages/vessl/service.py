import abc
import os
import pickle
import subprocess
import types
from io import BufferedReader
from typing import Any, Callable, Dict, List, TypeVar, Union

import cloudpickle
import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from schema import Or, Schema

from vessl.cli._util import Endpoint, format_url
from vessl.model import (
    create_model,
    download_model_volume_file,
    list_model_volume_files,
    read_model,
    upload_model_volume_file,
)
from vessl.organization import _get_organization_name
from vessl.util import logger
from vessl.util.common import get_module
from vessl.util.exception import (
    InvalidParamsError,
    VesslApiException,
    VesslRuntimeException,
)

ModelType = TypeVar("ModelType")
InputDataType = bytes
ModelInputDataType = TypeVar("ModelInputDataType")
ModelOutputDataType = TypeVar("ModelOutputDataType")
OutputDataType = TypeVar("OutputDataType")

PROTOCOL_VERSION = 4
USER_DEFINED = "user-defined"
MANIFEST_FILENAME = "vessl.manifest.yaml"
RUNNER_FILENAME = "vessl.runner.pkl"
DEFAULT_MODEL_FILENAME = "vessl.model.pkl"


class RunnerBase(abc.ABC):
    """Base class for model registering.

    This base class introduces 5 static methods as followings:
    - `predict`: Make prediction with given data and model. This method must be overridden. The
      data is given from the result of `preprocess_data`, and the return value of this method
      will be passed to `postprocess_data` before serving.
    - `save_model`: Save the model into a file. Return value of this method will be given to the
      `load_model` method on model loading. If this method is overriden, `load_model` must be
      overriden as well.
    - `load_model`: Load the model from a file.
    - `preprocess_data`: Preprocess the data before prediction. It converts the API input data to
      the model input data.
    - `postprocess_data`: Postprocess the data after prediction. It converts the model output data
      to the API output data.

    Check each method's docstring for more information.
    """

    @staticmethod
    def save_model(model: ModelType) -> Dict[str, str]:
        """Save the given model instance into file.

        Return value of this method will be given to first argument of `load_model` on model loading.

        Args:
            model(model_instance): Model instance to save.

        Returns:
            (dict) Data that will be passed to `load_model` on model loading.
                Must be a dictionary with key and value both string.
        """
        pass

    @staticmethod
    def load_model(props: Union[Dict[str, str], None], artifacts: Dict[str, str]) -> ModelType:
        """Load the model instance from file.

        `props` is given from the return value of `save_model`, and `artifacts` is
        given from the `register_model` method.

        If the `save_model` is not overriden, `props` will be None

        Args:
            props(dict | None): Data that was returned by `save_model`. If `save_model` is
                not overriden, this will be None.
            artifacts(dict): Data that is given by `register_model` function.

        Returns:
            Model instance.
        """
        pass

    @staticmethod
    def preprocess_data(data: InputDataType) -> ModelInputDataType:
        """Preprocess the given data.

        The data processed by this method will be given to the model.

        Args:
            data: Data to be preprocessed.

        Returns:
            Preprocessed data that will be given to the model.
        """
        pass

    @staticmethod
    def postprocess_data(data: ModelOutputDataType) -> OutputDataType:
        """Postprocess the given data.

        The data processed by this method will be given to the user.

        Args:
            data: Data to be postprocessed.

        Returns:
            Postprocessed data that will be given to the user.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def predict(model: ModelType, data: ModelInputDataType) -> ModelOutputDataType:
        """Make prediction with given data and model.

        Args:
            model(model_instance): Model instance.
            data: Data to be predicted.

        Returns:
            Prediction result.
        """
        raise NotImplementedError()


_manifest_schema = Schema(
    {
        "version": str,
        "model": {
            "save_type": Or(str, None),
            "save_path": Or(str, None),
            "save_output": Or({str: str}, None),
        },
        "runner": {
            "save_type": str,
            "save_path": str,
        },
        "use_default": [str],
        "requirements": [str],
        "artifacts": Or({str: str}, {}),
    }
)


def _save_object(save_type: Union[str, None], obj: Any, path: str) -> str:
    """Save the object into `path`

    If `save_type` is not `None`, use the given serialization method. It will not fallback on failure.
    Otherwise, use various methods to serialize the object as best as possible.
    After serialization, save it into `path`.

    Args:
        save_type (str | None): The serialization method.
        obj (Any): The object to serialize.
        path (str): The path to save the object.

    Returns:
        (str) The save type finally applied to the object.
    """
    # Known save types
    if save_type == "torch":
        import torch

        with open(path, "wb") as f:
            torch.save(obj, f, pickle_module=cloudpickle, pickle_protocol=PROTOCOL_VERSION)
        return "torch"
    if save_type is not None:
        # save_type given, but we don't know how to handle it
        raise InvalidParamsError("Unknown save type: {}".format(save_type))

    # Fallback saving methods for None save_type
    with open(path, "wb") as f:
        # Try cloudpickle first
        try:
            cloudpickle.dump(obj, f, protocol=PROTOCOL_VERSION)
            return "cloudpickle"
        except pickle.PicklingError:
            pass

        # Try joblib
        try:
            joblib = get_module("joblib")
            joblib.dump(obj, f)
            return "joblib"
        except pickle.PicklingError:
            pass

        # Last fallback. Use pickle module
        try:
            pickle.dump(obj, f, protocol=PROTOCOL_VERSION)
            return "pickle"
        except pickle.PicklingError:
            pass
    # Failed to save. Raise.
    raise VesslRuntimeException("Could not save the object")


def _try_load(loader: Callable, br: BufferedReader) -> Any:
    try:
        return loader(br)
    except Exception as e:
        raise VesslRuntimeException(
            f"Could not load the object. Did you use a different Python version to register the model?\nError: {e}"
        )


def _load_object(save_type: str, path: str) -> Any:
    """Load the object from given path

    It uses the given save_type to find a correct way to deserialize the object.

    Args:
        save_type (str): The serialization method used to save the object.
        path (str): The path to load the object.

    Returns:
        (Any) The loaded object.
    """
    if save_type == "torch":
        with open(path, "rb") as f:
            torch = get_module(
                "torch", required='torch package is required. Run "pip install torch".'
            )
            return _try_load(torch.load, f)

    with open(path, "rb") as f:
        if save_type == "cloudpickle":
            return _try_load(cloudpickle.load, f)
        if save_type == "joblib":
            joblib = get_module(
                "joblib",
                required='joblib package is required. Run "pip install joblib".',
            )
            return _try_load(joblib.load, f)
        if save_type == "pickle":
            return _try_load(pickle.load, f)
    raise InvalidParamsError("Unknown save type")


def _upload_artifacts(
    artifacts: Dict[str, str],
    *,
    repository_name: str,
    model_number: int,
    organization_name: str,
) -> None:
    """Upload given artifacts to the model volume.

    Args:
        artifacts(dict): Artifacts to be uploaded. Key is the path to artifact in local filesystem, and value is the
            path in the model volume. Only trailing asterisk(*) is allowed for glob pattern.
        repository_name(str): Name of the repository.
        model_number(int): Number of the model.
        organization_name(str): Name of the organization.

    Example:
        >>> _upload_artifacts({
        >>>     "./output/*": "*",
        >>>     "./sample.data": "sample.data",
        >>> }, repository_name="my-repository", model_number=1, organization_name="my-organization")
    """
    # Input Validation
    for local_path, remote_path in artifacts.items():
        local_dir, local_base = os.path.dirname(local_path), os.path.basename(local_path)
        remote_dir, remote_base = os.path.dirname(remote_path), os.path.basename(remote_path)
        if "*" in local_dir or "*" in remote_dir:
            raise InvalidParamsError("Only trailing asterisk(*) is allowed for glob pattern")
        if ("*" in local_base and local_base != "*") or ("*" in remote_base and remote_base != "*"):
            raise InvalidParamsError("Only trailing asterisk(*) is allowed for glob pattern")
        if (local_base == "*") != (remote_base == "*"):
            # Both should either or neither be glob pattern
            raise InvalidParamsError(
                "Both local and remote path should either or neither be glob pattern"
            )

    if len(artifacts.keys()) != 0:
        print("Artifacts will be uploaded as follows:")
        for l, r in artifacts.items():
            print("  {} -> {}".format(l, r))

    for local_path, remote_path in artifacts.items():
        local_dir, local_base = os.path.dirname(local_path), os.path.basename(local_path)
        remote_dir, remote_base = os.path.dirname(remote_path), os.path.basename(remote_path)
        if local_base == "*":
            # Glob pattern
            for root, _, files in os.walk(local_dir):
                for filename in files:
                    local_filepath = os.path.join(root, filename)
                    upload_model_volume_file(
                        repository_name=repository_name,
                        model_number=model_number,
                        organization_name=organization_name,
                        source_path=local_filepath,
                        dest_path=local_filepath.replace(local_dir, remote_dir),
                    )
        else:
            # Upload single file
            upload_model_volume_file(
                repository_name=repository_name,
                model_number=model_number,
                organization_name=organization_name,
                source_path=local_path,
                dest_path=remote_path,
            )


# Related to model registration. Will be used as Python SDK, not CLI.
def _register_serving_method(
    organization_name: str,
    repository_name: str,
    model_number: Union[int, None],
    model_type: Union[str, None],
    model_instance: Union[ModelType, None],
    runner_cls: RunnerBase,
    requirements: List[str],
    artifacts: Dict[str, str],
):
    # Argument validation
    # Valid comb of (model_number, model_type, model_instance) (0 means None)
    # - (0, 0, 1): Create new model from given model_instance and register. Default fallback saver will be used since model_type is not given.
    # - (0, 1, 1): Create new model from given model_instance and register. model_type's saver will be used.
    # - (1, 0, 0): Register existing model. It will only add runner/manifest file to make model servable.
    # - (1, 0, 1): Overwrite existing model. Given model_instance will be serialized and saved to given model number. Default fallback saver will be used since model_type is not given.
    # - (1, 1, 1): Overwrite existing model. Given model_instance will be serialized and saved to given model number. model_type's saver will be used.
    if model_number is None and model_instance is None:  # Exclude (0, x, 0)
        raise InvalidParamsError("`model_instance` must be given to create a new model")
    if model_instance is None and model_type is not None:  # Exclude (x, 1, 0)
        raise InvalidParamsError("`model_instance` must be given if `model_type` is not None")
    for key in artifacts:
        if key in [RUNNER_FILENAME, MANIFEST_FILENAME, DEFAULT_MODEL_FILENAME]:
            raise InvalidParamsError("`{}` is a reserved key of artifacts".format(key))

    if model_number is not None:
        # Check model existence
        try:
            read_model(
                repository_name=repository_name,
                model_number=model_number,
                organization_name=organization_name,
            )
        except VesslApiException:
            raise InvalidParamsError("Failed to read requested model")
    else:
        # Create new model
        logger.info("`model_number` is None. New model will be created.")
        res = create_model(
            repository_name=repository_name,
            organization_name=organization_name,
        )
        model_number = res.number

    # Runner class validation
    runner_cls()  # Emits error if abstract method is not overriden
    if model_instance is None:
        # Tries to add serving method only.
        if runner_cls.load_model == RunnerBase.load_model:
            raise InvalidParamsError(
                "`load_model` method must be overriden if model_instance is not given"
            )
    else:
        if (
            runner_cls.save_model != RunnerBase.save_model
            and runner_cls.load_model == RunnerBase.load_model
        ):
            raise InvalidParamsError(
                "`load_model` must be implemented if `save_model` is implemented"
            )
    # Validation done.

    # Business logic from here
    # Save the model
    model_save_type, model_save_path, model_save_output = USER_DEFINED, None, None
    if model_instance is not None:
        if runner_cls.save_model != RunnerBase.save_model:
            # Use the custom `save_model` if defined
            model_save_type = USER_DEFINED
            model_save_output = runner_cls.save_model(model_instance)
        elif runner_cls.load_model != RunnerBase.load_model:
            # `save_model` is not overriden but `load_model` is overridden.
            # We can assume user saves the model in other way. Skip saving
            model_save_type = USER_DEFINED
        else:
            # Use the default saver
            model_save_type = _save_object(model_type, model_instance, DEFAULT_MODEL_FILENAME)
            model_save_path = DEFAULT_MODEL_FILENAME

    # Save the runner
    runner_save_type, runner_save_path = "", ""
    runner_save_type = _save_object(None, runner_cls, RUNNER_FILENAME)
    runner_save_path = RUNNER_FILENAME

    # Mark the methods that are not overriden by user
    use_default = []
    if runner_cls.load_model == RunnerBase.load_model:
        use_default.append("load_model")
    if runner_cls.preprocess_data == RunnerBase.preprocess_data:
        use_default.append("preprocess_data")
    if runner_cls.postprocess_data == RunnerBase.postprocess_data:
        use_default.append("postprocess_data")

    # Save the manifest
    manifest = {
        "version": "v1",
        "model": {
            "save_type": model_save_type,
            "save_path": model_save_path,
            "save_output": model_save_output,
        },
        "runner": {
            "save_type": runner_save_type,
            "save_path": runner_save_path,
        },
        "use_default": use_default,
        "requirements": requirements,
        "artifacts": artifacts,
    }
    _manifest_schema.validate(manifest)
    with open(MANIFEST_FILENAME, "w") as f:
        yaml.dump(manifest, f)

    # Upload artifacts
    upload = artifacts.copy()
    upload[MANIFEST_FILENAME] = MANIFEST_FILENAME
    upload[runner_save_path] = runner_save_path
    if model_save_path is not None:
        upload[model_save_path] = model_save_path
    _upload_artifacts(
        upload,
        repository_name=repository_name,
        model_number=model_number,
        organization_name=organization_name,
    )

    print(
        f"Successfully registered model: {format_url(Endpoint.model.format(organization_name, repository_name, model_number))}"
    )


def register_model(
    repository_name: str,
    model_number: Union[int, None],
    runner_cls: RunnerBase,
    model_instance: Union[ModelType, None] = None,
    requirements: List[str] = None,
    artifacts: Dict[str, str] = None,
    **kwargs,
):
    """Register the given model for serving. If you want to override the
    default organization, then pass `organization_name` as `**kwargs`.

    Args:
        repository_name(str): Model repository name.
        model_number(int | None): Model number. If None, new model will be
            created. In such case, `model_instance` must be given.
        runner_cls(RunnerBase): Runner class that includes code for serving.
        model_instance(ModelType | None): Model instance. If None, `runner_cls`
            must override `load_model` method. Defaults to None.
        requirements(List[str]): Python requirements for the model. Defaults to
            [].
        artifacts(Dict[str, str]): Artifacts to be uploaded. Key is the path to
            artifact in local filesystem, and value is the path in the model
            volume. Only trailing asterisk(*) is allowed for glob pattern.
            Defaults to {}.

    Example:
        ```python
        register_model(
            repository_name="my-model",
            model_number=1,
            runner_cls=MyRunner,
            model_instance=model_instance,
            requirements=["torch", "torchvision"],
            artifacts={"model.pt": "model.pt", "checkpoints/*": "checkpoints/*"},
        )
        ```
    """
    organization_name = _get_organization_name(**kwargs)
    if requirements is None:
        requirements = []
    if artifacts is None:
        artifacts = {}
    _register_serving_method(
        organization_name=organization_name,
        repository_name=repository_name,
        model_number=model_number,
        model_type=None,
        model_instance=model_instance,
        runner_cls=runner_cls,
        requirements=requirements,
        artifacts=artifacts,
    )


def register_torch_model(
    repository_name: str,
    model_number: Union[int, None],
    model_instance: ModelType,
    preprocess_data=None,
    postprocess_data=None,
    requirements: List[str] = None,
    **kwargs,
):
    """Register the given torch model instance for model serving. If you want to
    override the default organization, then pass `organization_name` as
    `**kwargs`.

    Args:
        repository_name(str): Model repository name.
        model_number(int | None): Model number. If None, new model will be
            created.
        model_instance(model_instance): Torch model instance.
        preprocess_data(callable): Function that will preprocess data.
            Defaults to identity function.
        postprocess_data(callable): Function that will postprocess data.
            Defaults to identity function.
        requirements(list): List of requirements. Defaults to [].

    Example:
        ```python
        vessl.register_model(
            repository_name="my-model",
            model_number=1,
            model_instance=model_instance,
            preprocess_data=lambda x: int(x),
            postprocess_data=lambda x: {"prediction": x},
            requirements=["torch", "torchvision"],
        )
        ```
    """
    # Check if the model is torch
    torch_nn = get_module(
        "torch.nn", required='torch package is required. Run "pip install torch".'
    )
    if not isinstance(model_instance, torch_nn.Module):
        raise InvalidParamsError("model_instance must be a torch.nn.Module.")

    organization_name = _get_organization_name(**kwargs)
    if requirements is None:
        requirements = []

    class PyTorchRunner(RunnerBase):
        @staticmethod
        def predict(model: ModelType, data: ModelInputDataType) -> ModelOutputDataType:
            return model(data)

    if preprocess_data is not None:
        PyTorchRunner.preprocess_data = preprocess_data
    if postprocess_data is not None:
        PyTorchRunner.postprocess_data = postprocess_data
    _register_serving_method(
        organization_name=organization_name,
        repository_name=repository_name,
        model_number=model_number,
        model_type="torch",
        model_instance=model_instance,
        runner_cls=PyTorchRunner,
        requirements=requirements + ["torch", "torchvision"],
        artifacts={},
    )


# Related to model serving. Will be used by CLI.
class _ModelServer(object):
    """Model server class."""

    def __init__(self, model: ModelType, preprocess_data, predict, postprocess_data):
        self.model = model
        self.preprocess_data = preprocess_data
        self.predict = predict
        self.postprocess_data = postprocess_data

    def run(self, data: InputDataType) -> OutputDataType:
        processed = self.preprocess_data(data)
        result = self.predict(self.model, processed)
        return self.postprocess_data(result)


def _load_model(install_reqs: bool = False) -> _ModelServer:
    """Load model from file. This function assumes that cwd is the root of model volume.

    Args:
        install_reqs(bool): Whether to install requirements. Default is False.

    Returns:
        (ModelServer) Model server.
    """
    with open(MANIFEST_FILENAME, "r") as f:
        try:
            manifest = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError:
            raise VesslRuntimeException("Invalid manifest file. Please register the model again.")

    _manifest_schema.validate(manifest)
    # Check for requirements
    if install_reqs:
        reqs = manifest["requirements"]
        if reqs:
            logger.info("Installing requirements:", reqs)
            subprocess.check_call(["pip", "install", *reqs])  # Support for conda?

    # Load runner
    runner_cls = _load_object(manifest["runner"]["save_type"], manifest["runner"]["save_path"])
    # Load model
    if "load_model" in manifest["use_default"]:
        model = _load_object(manifest["model"]["save_type"], manifest["model"]["save_path"])
    else:  # Use custom loader
        model = runner_cls.load_model(manifest["model"]["save_output"], manifest["artifacts"])
    # Load methods
    preprocess_data = (
        (lambda x: x)
        if "preprocess_data" in manifest["use_default"]
        else runner_cls.preprocess_data
    )
    postprocess_data = (
        (lambda x: x)
        if "postprocess_data" in manifest["use_default"]
        else runner_cls.postprocess_data
    )
    model_server = _ModelServer(model, preprocess_data, runner_cls.predict, postprocess_data)
    return model_server


def _run_model_server(model: _ModelServer, remote: bool):
    """Boot simple server that serves the model."""
    app = FastAPI()

    def verify_key(request: Request):
        if not remote:
            return True

        token = request.headers.get("X-AUTH-KEY")
        if token is None or token != os.environ.get("SERVING_AUTH_KEY"):
            raise HTTPException(status_code=401, detail="Invalid auth key.")
        return True

    @app.post("/")
    async def index(request: Request, authorized: bool = Depends(verify_key)):
        if authorized:
            body = await request.body()
            print("Received:", body)
            result = model.run(body)
            print("Result:", result)
            return result

    uvicorn.run(app, host="0.0.0.0", port=8000)


def serve_model(
    repository_name: str,
    model_number: int,
    install_reqs: bool = False,
    remote: bool = False,
    **kwargs,
):
    """Execute the server which serves the given model. If you want to override the default organization, then pass `organization_name` as `**kwargs`.

    This method will create `.vessl-serving` under current directory to save the artifacts in local.

    Args:
        repository_name(str): Model repository name.
        model_number(int): Model number.
        install_reqs(bool): Whether to install requirements. Defaults to False.
    """
    # Validation
    organization_name = _get_organization_name(**kwargs)
    model = read_model(repository_name=repository_name, model_number=model_number, **kwargs)
    if not model.is_servable:
        raise InvalidParamsError("Model is not servable.")

    if not remote:
        print("Downloading model artifacts for serving...")
        # Create temporal directory if not exists
        local_path_root = f".vessl-serving/{organization_name}/{repository_name}/{model_number}"
        os.makedirs(local_path_root, exist_ok=True)

        # Get the list of files to be downloaded. Only the outdated files or the files that are not in the local will be downloaded.
        volume_files = list_model_volume_files(
            repository_name=repository_name, model_number=model_number, recursive=True
        )
        local_status_filename = (
            ".vessl.localstatus.yaml"  # File that keeps track of the local status of the files
        )
        local_status = {  # Structure of the local status file
            "hash": {},
        }
        download_list = []
        if os.path.exists(os.path.join(local_path_root, local_status_filename)):
            # Local status file exists -> do not download the files that are already up-to-date.
            with open(os.path.join(local_path_root, local_status_filename), "r") as f:
                local_status.update(yaml.load(f, Loader=yaml.FullLoader))
            for volume_file in volume_files:
                if volume_file.is_dir:
                    continue  # Skip directory
                if volume_file.hash == local_status.get("hash", {}).get(volume_file.path, None):
                    continue  # Skip if hash is up to date
                download_list.append(volume_file)
        else:
            # Local status file does not exist -> download all files.
            for volume_file in volume_files:
                if volume_file.is_dir:
                    continue
                download_list.append(volume_file)
        # Download list retrieved.

        # Download files
        os.chdir(local_path_root)  # Make root of the model volume as cwd
        for volume_file in download_list:
            download_model_volume_file(
                repository_name=repository_name,
                model_number=model_number,
                source_path=volume_file.path,
                dest_path=volume_file.path,
            )
            # Update hash so that we don't download the same file again.
            local_status["hash"][volume_file.path] = volume_file.hash
        with open(local_status_filename, "w") as f:
            yaml.dump(local_status, f)
        print("Artifacts for model serving are up to date.")

    # Serve model
    print("Loading model from artifacts...")
    model = _load_model(install_reqs)
    print("Booting local server for serving...")
    _run_model_server(model, remote)
