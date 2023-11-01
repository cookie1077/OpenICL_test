import os.path
import subprocess
from typing import List

from vessl.util.vesslignore import list_files_using_vesslignore


class Tar:
    @staticmethod
    def gzip(output_file_name: str, path_to_zip: str, exclude_paths: List = ()) -> int:
        """
        exclude_paths: relative path from path_to_zip
        """
        commands = ["tar", "-C", path_to_zip]
        for exclude_path in exclude_paths:
            commands.extend(["--exclude", exclude_path])

        commands.extend(["-zcf", output_file_name, "."])

        subprocess.run(commands).check_returncode()
        return os.path.getsize(output_file_name)

    @staticmethod
    def gzip_using_vesslignore(
        output_file_name: str,
        path_to_zip: str,
        exclude_paths: List = (),
        vesslignore_filename: str = ".vesslignore",
    ) -> int:
        """
        Make a tarball of the directory `path_to_zip`, using vesslignore pattern file.

        The files are recursively listed and filtered by vesslignore logic.
        The resulting list is supplied to tar via standard input (using the command line parameter '-X').
        """
        commands = ["tar", "-c"]
        commands.extend(["-z"])  # gzip
        commands.extend(
            ["--no-recursion"]
        )  # we choose which files to add; do not add all files when given a directory
        for exclude_path in exclude_paths:
            commands.extend(["--exclude", exclude_path])
        commands.extend(["-C", path_to_zip])  # move base directory
        commands.extend(["-f", output_file_name])  # output file
        commands.extend(["-T", "-"])  # take the list of files to include from stdin

        all_files_to_include = list_files_using_vesslignore(
            path_to_zip, vesslignore_filename=vesslignore_filename
        )

        subprocess.run(
            commands, input="\n".join(all_files_to_include), encoding="utf-8"
        ).check_returncode()

        return os.path.getsize(output_file_name)

    @staticmethod
    def extract(tar_file_name: str, path_to_extract: str):
        subprocess.run(["mkdir", "-p", path_to_extract]).check_returncode()
        subprocess.run(["tar", "xf", tar_file_name, "-C", path_to_extract]).check_returncode()
