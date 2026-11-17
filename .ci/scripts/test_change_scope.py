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

"""Tests for .ci/scripts/change_scope.py, the Make/Bazel job selector."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent / "change_scope.py"
_spec = importlib.util.spec_from_file_location("change_scope", _SCRIPT)
assert _spec and _spec.loader
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

BOTH = (True, True)
MAKE = (True, False)
BAZEL = (False, True)


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        # What Renovate changes on its own.
        (["MODULE.bazel"], BAZEL),
        (["MODULE.bazel", "MODULE.bazel.lock"], BAZEL),
        ([".bazelversion"], BAZEL),
        (["dev/bazel/deps/requirements_lock.txt"], BAZEL),
        (["dev/bazel/deps/gcc.bzl"], BAZEL),
        ([".ci/env/bazelisk.sh", ".ci/env/bazelisk.ps1"], BAZEL),
        ([".ci/env/apt.sh", "dev/bazel/deps/requirements_lock.txt"], BOTH),
        ([".ci/env/tbb.sh", ".ci/pipeline/ci.yml", "dev/download_tbb.sh"], BOTH),
        # Build files anywhere in the tree.
        (["cpp/oneapi/dal/algo/kmeans/BUILD", "dev/bazel/dal.bzl"], BAZEL),
        (["examples/oneapi/cpp/BUILD.bazel"], BAZEL),
        (["makefile", "makefile.lst", "dev/make/compiler_definitions/gnu.mk"], MAKE),
        ([".ci/scripts/build.sh", ".ci/scripts/build.bat", ".ci/scripts/test.bat"], MAKE),
        # Read by both systems, or by neither: everything runs.
        (["MODULE.bazel", "makefile"], BOTH),
        (["makefile.ver"], BOTH),
        ([".ci/scripts/test.sh"], BOTH),
        (["cmake/templates/oneDALConfig.cmake.in"], BOTH),
        (["deploy/local/vars_lnx.sh"], BOTH),
        (["cpp/daal/src/algorithms/kmeans/kmeans_impl.i"], BOTH),
        ([".github/workflows/ci.yml"], BOTH),
        (["dev/bazel/README.md", "cpp/oneapi/dal/BUILD"], BAZEL),
        ([], BOTH),
    ],
)
def test_scope(paths, expected):
    assert cs.scope(paths) == expected


def test_a_name_that_only_contains_build_is_not_a_build_file():
    assert cs.scope(["cpp/daal/src/BUILDING.txt"]) == BOTH


def test_cli_prints_github_output_lines(capsys):
    assert cs.main(["MODULE.bazel"]) == 0
    assert capsys.readouterr().out == "make=false\nbazel=true\n"
    assert cs.main([r"dev\make\deps.mk"]) == 0
    assert capsys.readouterr().out == "make=true\nbazel=false\n"


def test_cli_needs_exactly_one_input_form():
    with pytest.raises(SystemExit):
        cs.main([])
    with pytest.raises(SystemExit):
        cs.main(["--base", "HEAD", "MODULE.bazel"])


def test_cli_diffs_two_commits(tmp_path, monkeypatch):
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q", "-b", "main", tmp_path], check=True)
    repo = tmp_path

    def commit(name: str, msg: str) -> None:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(msg + "\n", encoding="utf-8")
        subprocess.run([*git, "-C", repo, "add", "-A"], check=True)
        subprocess.run([*git, "-C", repo, "commit", "-q", "-m", msg], check=True)

    commit("README.md", "base")
    commit("MODULE.bazel", "bump")
    monkeypatch.setattr(cs, "REPO", repo)
    assert cs.changed_files("HEAD^1", "HEAD") == ["MODULE.bazel"]
    assert cs.main(["--base", "HEAD^1", "--head", "HEAD"]) == 0
    commit("dev/make/deps.mk", "make")
    assert cs.scope(cs.changed_files("HEAD~2", "HEAD")) == BOTH
    assert cs.main(["--base", "no-such-revision"]) == 2
