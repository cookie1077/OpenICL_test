import os
import pathlib
import re
from typing import List, Union

from vessl.util import logger


class InvalidPatternException(Exception):
    pass


# Regex that matches all parent directories ("ab/cd/ef/", for example)
REGEX_ALL_DIRS_PREFIX = r"([^/]+/)*"


def _segment_to_regex(segment: str) -> str:
    """
    Given a pattern for a path segment (i.e. a name of a directory or a file), returns a regex
    pattern that matches the segment exactly.

    It supports `*` (matches zero or more characters) and `?` (matches exactly one character).
    These wildcards do not match directory separator (`/`).
    """
    translate_map = {i: "\\" + chr(i) for i in b"()[]{}+-|^$\\.&~# \t\n\r\v\f"}
    # translate takes keys as integers only
    translate_map[b"?"[0]] = "[^/]"
    translate_map[b"*"[0]] = "[^/]*"
    return segment.translate(translate_map)


class Pattern(object):
    """
    Pattern parses and holds a single line in vesslignore file.

    Args:
        pattern (str): A pattern string in .vesslignore file.

    Attributes:
        _regex (re.Pattern): Parsed pattern in regex.
            A path can be checked by _regex.match() (match from the beginning).
    """

    _regex: re.Pattern

    def __init__(self, pattern: str):
        self._parse(pattern)
        logger.debug(f"Parsed pattern {pattern} into regex {self._regex.pattern}")

    def __str__(self) -> str:
        return f"<Pattern: regex {self._regex.pattern}>"

    def __repr__(self) -> str:
        return f"<Pattern: regex {self._regex.pattern}>"

    def _parse(self, pattern: str):
        if self._try_parse_simple(pattern):
            return

        if self._try_parse_all_dirs_prefixed(pattern):
            return

        if self._try_parse_relative(pattern):
            return

        raise InvalidPatternException(pattern)

    def _try_parse_simple(self, pattern: str) -> bool:
        """
        Try to parse a pattern string as a simple field. Both files and directories can match
        against this pattern.

        It supports `*` (matches zero or more characters) and `?` (matches exactly one character).
        These wildcards do not match against directory separator (`/`).

        Examples include:
        - `db.sqlite3` (all files or directories with this name in all subdirectories will match)
        - `*.pyc`
        - `.*.sw?` (example for Vim swap-files)
        - `__pycache__` (note that directories also match this pattern)

        Args:
            pattern (str): A pattern string in .vesslignore file.

        Returns:
            True if parse has succeeded.
        """

        PATTERN_REGEX_SIMPLE = r"[^/]+"
        if re.fullmatch(PATTERN_REGEX_SIMPLE, pattern) is None:
            return False

        regex = REGEX_ALL_DIRS_PREFIX + _segment_to_regex(pattern)

        regex += r"(/|$)"  # match file ($) or directory (/)

        self._regex = re.compile(regex)
        return True

    def _try_parse_all_dirs_prefixed(self, pattern: str) -> bool:
        """
        Try to parse a pattern string with preceding `**/`. Both files and directories can match
        against this pattern; a trailing directory separator (`/`) will force a directory match.

        The remaining part of the pattern should not contain neither of `*`, `?`, nor `//`, and
        denotes a pattern that matches against any file or directory in the tree.

        Examples include:
        - `**/bin/Debug/` (matches all `Debug` directories that are under `bin`)
        - `**/nbproject/private` (matches all files or directories that are named `private`
          and are under `nbproject`)

        Args:
            pattern (str): A pattern string in .vesslignore file.

        Returns:
            True if parse has succeeded.
        """
        PATTERN_REGEX_ALL_DIRS_PREFIX = r"\*\*/[^*?]+"
        if re.fullmatch(PATTERN_REGEX_ALL_DIRS_PREFIX, pattern) is None:
            return False
        if "//" in pattern or "**" in pattern[1:]:
            return False

        regex = REGEX_ALL_DIRS_PREFIX
        pattern_body = pattern[3:]
        if "*" in pattern_body or "?" in pattern_body:
            return False

        trailing_slash = False
        if pattern_body.endswith("/"):
            trailing_slash = True
            pattern_body = pattern_body[:-1]
            if not pattern_body:
                # "**//" case
                return False

        path_segments = pattern_body.split("/")
        regex += r"/".join(map(_segment_to_regex, path_segments))

        # At this point, pattern `**/a/b` may match against path `a/bc`;
        # we should add a delimiter. Use trailing_slash to determine which delimiter to use.
        regex += r"/" if trailing_slash else r"(/|$)"

        self._regex = re.compile(regex)
        return True

    def _try_parse_relative(self, pattern: str) -> bool:
        """
        Try to parse a pattern string that matches relatively to current directory.
        Both files and directories can match against this pattern; a trailing directory separator
        (`/`) will force a directory match.

        It supports `*` (matches zero or more characters) and `?` (matches exactly one character).
        These wildcards do not match against directory separator.

        The pattern string should not contain `**` or `//`. Leading directory separator is ignored.

        Note:
            The pattern should contain at least one directory separator character to match this case;
            otherwise, it would be considered as a 'simple' pattern (like `*.so`).

        Examples include:
        - `.venv/` (matches `.venv` directory at the same level as this .vesslignore file)
        - `*.egg-info/`
        - `` (may match file or directory)

        Args:
            pattern (str): A pattern string in .vesslignore file.

        Returns:
            True if parse has succeeded.
        """
        if "**" in pattern or "//" in pattern:
            return False

        if "/" not in pattern:
            return False

        trailing_slash = pattern.endswith("/")
        pattern = pattern.strip("/")  # strip both leading and trailing slash
        if not pattern:
            # "/" case
            return False

        path_segments = pattern.split("/")
        regex = r"/".join(map(_segment_to_regex, path_segments))

        # At this point, pattern `a/b` may match against path `a/bc`;
        # we should add a delimiter. Use trailing_slash to determine which delimiter to use.
        regex += r"/" if trailing_slash else r"(/|$)"

        self._regex = re.compile(regex)
        return True

    def matches(self, path: str) -> bool:
        return self._regex.fullmatch(path) is not None
