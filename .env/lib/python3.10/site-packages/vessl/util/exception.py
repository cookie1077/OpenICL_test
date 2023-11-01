import json

from openapi_client.exceptions import ApiException
from vessl._version import __VERSION__
from vessl.util import logger

## Error Codes
DEFAULT_ERROR_CODE = "UnexpectedProblem"
ACCESS_TOKEN_NOT_FOUND_ERROR_CODE = "ACCESS_TOKEN_NOT_FOUND"

## Error Messages
DEFAULT_ERROR_MESSAGE = (
    "An unexpected exception occurred. Use VESSL_LOG=DEBUG to view stack trace. "
    "(CLI version: %s)\n"
    "(`pip install --upgrade vessl` might resolve this issue.)" % __VERSION__
)


class VesslException(Exception):
    def __init__(self, message=DEFAULT_ERROR_MESSAGE, code=DEFAULT_ERROR_CODE, exit_code=1):
        self.code = code
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)


class VesslApiException(VesslException):
    def __init__(
        self, message=DEFAULT_ERROR_MESSAGE, code=DEFAULT_ERROR_CODE, exit_code=1, status=0
    ):
        super().__init__(message, code, exit_code)
        self.status = status

    @classmethod
    def convert_api_exception(cls, api_exception: ApiException) -> "VesslApiException":
        try:
            body = json.loads(api_exception.body)
        except json.JSONDecodeError:
            # In some case, the error response body is not JSON.
            # For example, when the server is down, the response body is HTML.
            logger.info("Failed to parse error response body.")
            return cls(code=api_exception.status, message=api_exception.reason)

        body_code = body.get("code", DEFAULT_ERROR_CODE)
        body_message = body.get("message", "")
        message = (
            f"{body_code} ({api_exception.status})" f"{': ' + body_message if body_message else ''}"
        )

        fields = body.get("fields")
        if fields:
            additional_messages = []
            for field in fields:
                field_name = field.get("name", "")
                field_value = field.get("value", "")
                field_message = field.get("message")
                additional_messages.append(
                    f"{field_name}: {field_value}"
                    f"{'(' + field_message + ')' if field_message else ''}"
                )
            message += f" {', '.join(additional_messages)}."

        return cls(code=body_code, message=message, status=api_exception.status)


class VesslRuntimeException(VesslException):
    pass


class ClusterAlreadyExistsError(VesslException):
    pass


class ClusterNotFoundError(VesslException):
    pass


class DependencyError(VesslException):
    pass


class GitError(VesslException):
    pass


class TimeoutError(VesslException):
    pass


class InvalidDatasetError(VesslException):
    pass


class InvalidKernelClusterError(VesslException):
    pass


class InvalidKernelImageError(VesslException):
    pass


class InvalidKernelResourceSpecError(VesslException):
    pass


class InvalidOrganizationError(VesslException):
    pass


class InvalidExperimentError(VesslException):
    pass


class InvalidProjectError(VesslException):
    pass


class InvalidTokenError(VesslException):
    pass


class InvalidVolumeFileError(VesslException):
    pass


class InvalidVolumeMountError(VesslException):
    pass


class InvalidWorkspaceError(VesslException):
    pass


class InvalidServingError(VesslException):
    pass


class InvalidParamsError(VesslException):
    pass


class InvalidTypeError(VesslException):
    pass


class InvalidYAMLError(VesslException):
    pass


class ImportPackageError(VesslException):
    pass


class ContextVariableNotFoundError(VesslException):
    pass


class NotFoundError(VesslException):
    pass


class BadRequestError(VesslException):
    pass


class UnexpectedProblemError(VesslException):
    pass
