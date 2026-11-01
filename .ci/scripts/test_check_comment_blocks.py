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

"""Tests for .ci/scripts/check_comment_blocks.py, the comment block length gate."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent / "check_comment_blocks.py"
_spec = importlib.util.spec_from_file_location("check_comment_blocks", _SCRIPT)
assert _spec and _spec.loader
ccb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccb)

LONG = "# one\n# two\n# three\nx = 1\n"
LONG_CPP = "// one\n// two\n// three\nint x = 1;\n"


@pytest.mark.parametrize(
    ("path", "mark"),
    [
        ("cpp/oneapi/dal/algo/knn/detail/infer_ops.hpp", "//"),
        ("cpp/daal/src/algorithms/kmeans/kmeans_impl.i", "//"),
        ("cpp/oneapi/dal/backend/primitives/kernel.cl", "//"),
        ("dev/release_tests/compare_release_trees.py", "#"),
        (".ci/scripts/build.sh", "#"),
        (".github/workflows/ci.yml", "#"),
        ("cpp/oneapi/dal/BUILD", "#"),
        ("dev/bazel/dal.bzl", "#"),
        ("makefile.lst", "#"),
        ("dev/make/compiler_definitions/gnu.mk", "#"),
        ("cmake/templates/oneDALConfig.cmake.in", None),
        ("docs/source/index.rst", None),
        ("README.md", None),  # `#` is a heading there
    ],
)
def test_which_files_are_read(path, mark):
    assert ccb.marker(path) == mark


def test_a_hash_inside_a_python_string_is_not_a_comment():
    text = 'x = """\n# a\n# b\n# c\n"""\n'
    assert ccb.comment_lines(text, "f.py") == set()


def test_a_trailing_comment_does_not_join_a_block():
    assert ccb.blocks(ccb.comment_lines("// a\n// b\nint x; // c\n", "f.cpp")) == [(1, 2)]
    assert ccb.blocks(ccb.comment_lines("# a\n# b\nx = 1  # c\n", "f.py")) == [(1, 2)]


def test_shebang_doxygen_and_pragmas_do_not_count():
    text = "#!/bin/bash\n# a\n# b\n# shellcheck disable=SC2086\n# c\n"
    assert ccb.blocks(ccb.comment_lines(text, "f.sh")) == [(2, 3), (5, 5)]
    cpp = "/// a\n/// b\n/// c\n// clang-format off\n//! d\n// NOLINT\n"
    assert ccb.comment_lines(cpp, "f.hpp") == set()


def test_block_comments_are_not_read():
    text = "/*\n * a\n * b\n * c\n */\n"
    assert ccb.long_blocks(text, "f.cpp", {1, 2, 3, 4, 5}) == []


def test_a_license_header_is_skipped():
    hash_header = "#" + "=" * 20 + "\n# Copyright contributors to the oneDAL project\n#\n"
    hash_header += "# Licensed under the Apache License\n# more\n#" + "=" * 20 + "\n"
    assert ccb.long_blocks(hash_header, "f.yml", set(range(1, 7))) == []
    cpp_header = "// Copyright contributors\n// a\n// b\n"
    assert ccb.long_blocks(cpp_header, "f.cpp", {1, 2, 3}) == []


def test_python_that_does_not_tokenize_falls_back_to_lines():
    assert ccb.comment_lines("# a\n# b\n# c\nx = '\n", "f.py") == {1, 2, 3}


def test_yaml_comments_are_read_by_line_including_in_run_scripts():
    text = "steps:\n  - run: |\n      # a\n      # b\n      # c\n      make\n"
    assert ccb.blocks(ccb.comment_lines(text, "f.yml")) == [(3, 5)]


def test_a_form_feed_keeps_line_numbers_in_step_with_git():
    assert ccb.long_blocks("a: 1\f\n# x\n# y\n# z\n", "f.yml", {2}) == [(2, 4, 3)]


def test_a_banner_is_one_line_but_rulers_do_not_hide_a_long_block():
    banner = "// ------\n// Section\n// ------\nint x;\n"
    assert ccb.long_blocks(banner, "f.cpp", {2}) == []
    ruled = "// ======\n// a\n// b\n// c\n// ======\n"
    assert ccb.long_blocks(ruled, "f.cpp", {3}) == [(1, 5, 3)]
    slashes = "// a\n// b\n" + "/" * 20 + "\n// c\n// d\n"
    assert ccb.long_blocks(slashes, "f.cpp", {1}) == [(1, 5, 4)]


@pytest.mark.parametrize(
    ("changed", "flagged"),
    [
        ({2}, [(1, 3, 3)]),  # an edited line inside the block
        ({4}, []),  # only the code below it changed
        (set(), []),  # an existing block nobody touched
    ],
)
@pytest.mark.parametrize(("text", "path"), [(LONG, "f.py"), (LONG_CPP, "f.cpp")])
def test_a_long_block_fails_only_when_the_pr_touches_it(text, path, changed, flagged):
    assert ccb.long_blocks(text, path, changed) == flagged


def test_two_lines_pass_even_when_both_are_new():
    assert ccb.long_blocks("// one\n// two\nint x;\n", "f.cpp", {1, 2}) == []


def _git_repo(path: Path) -> None:
    for cmd in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "t"],
        ["config", "commit.gpgsign", "false"],
    ):
        subprocess.run(["git", *cmd], cwd=path, check=True)


def _commit(path: Path, msg: str, tag: str | None = None) -> None:
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=path, check=True)
    if tag:
        subprocess.run(["git", "tag", tag], cwd=path, check=True)


def test_main_flags_a_new_long_block_with_its_lines(tmp_path, monkeypatch, capsys):
    _git_repo(tmp_path)
    (tmp_path / "a.cpp").write_text("int y;\n", encoding="utf-8")
    _commit(tmp_path, "base", tag="base")
    (tmp_path / "a.cpp").write_text("int y;\n" + LONG_CPP, encoding="utf-8")
    _commit(tmp_path, "pr")
    monkeypatch.setattr(ccb, "REPO", tmp_path)
    assert ccb.main(["--base", "base", "--head", "HEAD"]) == 1
    assert "::error file=a.cpp,line=2,endLine=4," in capsys.readouterr().out


def test_main_passes_an_untouched_long_block_and_a_renamed_file(tmp_path, monkeypatch):
    _git_repo(tmp_path)
    (tmp_path / "old.hpp").write_text(LONG_CPP + "int y;\n" * 10, encoding="utf-8")
    (tmp_path / "b.yml").write_text(LONG, encoding="utf-8")
    _commit(tmp_path, "base", tag="base")
    (tmp_path / "old.hpp").rename(tmp_path / "new.hpp")
    (tmp_path / "b.yml").write_text(LONG + "z: 3\n", encoding="utf-8")
    _commit(tmp_path, "move and append")
    monkeypatch.setattr(ccb, "REPO", tmp_path)
    assert ccb.findings("base", "HEAD") == []


def test_main_exits_2_for_a_base_git_cannot_diff(tmp_path, monkeypatch, capsys):
    _git_repo(tmp_path)
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    _commit(tmp_path, "base")
    monkeypatch.setattr(ccb, "REPO", tmp_path)
    assert ccb.main(["--base", "no-such-ref", "--head", "HEAD"]) == 2
    assert "no-such-ref" in capsys.readouterr().err
