#!/usr/bin/env python
# ===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

"""Report which build systems a change can affect, so CI runs only their jobs.

Prints ``make=<true|false>`` and ``bazel=<true|false>``, one per line, in the
``$GITHUB_OUTPUT`` format. A side is ``false`` only when every changed file is
one that the other build system alone reads: ``MODULE.bazel``, ``BUILD`` and
``.bzl`` files or ``dev/bazel/`` for Bazel, ``makefile`` or ``dev/make/`` for
Make. Any other file, such as a source, a shared dependency installer in
``.ci/env`` or a workflow, turns both sides on, as does an empty diff.

    python .ci/scripts/change_scope.py --base HEAD^1 --head HEAD
    python .ci/scripts/change_scope.py MODULE.bazel dev/bazel/deps/gcc.bzl

In a pull request build ``HEAD`` is the merge commit, so ``HEAD^1`` is the
target branch. Exits 2 when git cannot diff ``--base``/``--head``; the
workflows then run every job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Files only Bazel reads: exact paths, directory prefixes and file names.
BAZEL_ONLY_PATHS = frozenset(
    {
        "MODULE.bazel",
        "MODULE.bazel.lock",
        ".bazelversion",
        ".bazelrc",
        ".bazelignore",
        ".ci/env/bazelisk.sh",
        ".ci/env/bazelisk.ps1",
    }
)
BAZEL_ONLY_DIRS = ("dev/bazel/",)
BAZEL_ONLY_NAMES = frozenset({"BUILD", "BUILD.bazel"})
BAZEL_ONLY_SUFFIXES = (".bzl",)

# Files only Make reads. .ci/scripts/test.sh also runs the Bazel release tests.
MAKE_ONLY_PATHS = frozenset(
    {
        "makefile",
        "makefile.lst",
        ".ci/scripts/build.sh",
        ".ci/scripts/build.bat",
        ".ci/scripts/test.bat",
    }
)
MAKE_ONLY_DIRS = ("dev/make/",)


def is_bazel_only(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path in BAZEL_ONLY_PATHS
        or path.startswith(BAZEL_ONLY_DIRS)
        or name in BAZEL_ONLY_NAMES
        or name.endswith(BAZEL_ONLY_SUFFIXES)
    )


def is_make_only(path: str) -> bool:
    return path in MAKE_ONLY_PATHS or path.startswith(MAKE_ONLY_DIRS)


def scope(paths: list[str]) -> tuple[bool, bool]:
    """(make, bazel): whether the changed ``paths`` can affect each build system."""
    make = bazel = False
    for path in paths:
        if is_bazel_only(path):
            bazel = True
        elif is_make_only(path):
            make = True
        else:
            return True, True
    if not paths:
        return True, True
    return make, bazel


def changed_files(base: str, head: str) -> list[str]:
    """Paths that differ between ``base`` and ``head``; raises on a git error."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", base, head],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="git revision the change is compared against")
    parser.add_argument("--head", default="HEAD", help="git revision of the change")
    parser.add_argument("paths", nargs="*", help="changed files, instead of a git diff")
    args = parser.parse_args(argv)
    if bool(args.base) == bool(args.paths):
        parser.error("give either --base or a list of paths")
    if args.base:
        try:
            paths = changed_files(args.base, args.head)
        except subprocess.CalledProcessError as error:
            print(f"git diff failed: {error.stderr.strip()}", file=sys.stderr)
            return 2
    else:
        paths = [path.replace("\\", "/") for path in args.paths]
    make, bazel = scope(paths)
    print(f"make={str(make).lower()}")
    print(f"bazel={str(bazel).lower()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
