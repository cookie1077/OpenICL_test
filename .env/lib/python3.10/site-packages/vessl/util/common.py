import platform
from datetime import datetime, timezone
from importlib import import_module

import shortuuid
import timeago
from packaging.version import Version

from vessl.util import logger
from vessl.util.exception import ImportPackageError


def parse_time_to_ago(dt: datetime):
    if not dt:
        return "N/A"
    return timeago.format(dt, datetime.now(timezone.utc))


def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def get_module(name, required=None):
    try:
        return import_module(name)
    except ImportError:
        msg = f"Error importing optional module {name}"
        if required:
            logger.warn(msg)
            raise ImportPackageError(f"{required}")


def generate_uuid():
    generated_uuid = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return generated_uuid.random(8)


def is_arm_processor():
    required_version = Version("3.9.1")
    current_version = Version(platform.python_version())

    if current_version < required_version:
        # Below python 3.9.1, platform.processor() returns incorrect value on ARM
        # https://stackoverflow.com/questions/66842004/get-the-processor-type-using-python-for-apple-m1-processor-gives-me-an-intel-pro
        # TODO(mika): When we deprecate python 3.8, we can remove this check.
        from cpuinfo import get_cpu_info

        return (
            "m1" in get_cpu_info().get("brand_raw").lower()
            or "m2" in get_cpu_info().get("brand_raw").lower()
        )
    else:
        return platform.processor() == "arm"


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text
