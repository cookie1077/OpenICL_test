import os
import pathlib
from typing import List, Union

from vessl.util.vesslignore_pattern import Pattern


def _parse_file_patterns(filename: str) -> List[Pattern]:
    patterns: List[Pattern] = []

    with open(file=filename, newline="\n") as f:
        for line in f:
            line = line.strip()
            # Ignore empty or comment lines
            if len(line) == 0 or line.startswith("#"):
                continue

            patterns.append(Pattern(line))

    return patterns


def _check_all_matches(
    basename: str, dir_names: List[str], patterns: List[List[Pattern]], is_dir: bool
) -> bool:
    """
    Given a file and list of patterns in parent directories, check if there were
    any matching patterns.
    """
    path = (basename + "/") if is_dir else basename
    while patterns:
        for pattern in patterns[-1]:
            if pattern.matches(path):
                return True
        patterns = patterns[:-1]
        if patterns:
            path = dir_names[-1] + "/" + path
            dir_names = dir_names[:-1]
    return False


def list_files_using_vesslignore(
    scan_root: str, follow_symlinks: bool = True, vesslignore_filename: str = ".vesslignore"
) -> List[str]:
    files: List[str] = []

    def _list_files_recurse(
        target: Union[os.DirEntry, pathlib.Path],
        dir_names: List[str],
        patterns: List[List[Pattern]],
        is_root: bool,
    ):
        """
        Args:
            target:
                Current directory or file.

            dir_names:
                List of names of parent directories.

            patterns:
                List of .vesslignore patterns for each of the parent directories. For directories without
                .vesslignore files, an empty list should still be provided.

            is_root: Whether this is the scan root. Needed to drop the root directory from resulting paths.
        """

        is_dir = (
            target.is_dir(follow_symlinks=follow_symlinks)
            if isinstance(target, os.DirEntry)
            else target.is_dir()
        )
        basename = target.name or "."

        if not is_root:
            if _check_all_matches(basename, dir_names, patterns, is_dir):
                # This file or directory should be ignored. Stop here.
                return
            # Use relative path to scan_root (not the full path that contains scan_root),
            # so that we can use `tar -C`.
            relative_path = os.path.join(*dir_names, basename)
            files.append(relative_path)

        if is_dir:
            # Pick vesslignore entry from sub-entries of this directory.
            sub_entries = list(os.scandir(target))
            vesslignore_entry = next(
                (entry for entry in sub_entries if entry.name == vesslignore_filename), None
            )

            # Add patterns from vesslignore file (if exists).
            new_patterns = []
            if vesslignore_entry is not None and vesslignore_entry.is_file():
                new_patterns = _parse_file_patterns(vesslignore_entry.path)
            patterns = patterns + [new_patterns]

            if not is_root:
                dir_names = dir_names + [basename]

            for sub_entry in sub_entries:
                _list_files_recurse(sub_entry, dir_names, patterns, False)

    _list_files_recurse(pathlib.Path(scan_root), [], [], True)

    return files
