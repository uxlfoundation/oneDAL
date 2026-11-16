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

"""Fail a PR that adds or edits a line of a comment block longer than two lines.

A comment states the constraint or the reason in at most two lines; the
backstory goes in the PR or an issue, linked from the comment (``AGENTS.md``).
A block is a run of consecutive full-line comments: ``//`` in C, C++ and OpenCL
sources, ``#`` in Python, shell, YAML, TOML, CMake, Bazel and make files. Python
comments come from the tokenizer, so a ``#`` line inside a string is not one. A
block of three or more lines fails when the PR adds or changes any of its lines,
so an existing long block passes until a PR edits it.

Not counted, and ending a block: a shebang on line 1, Doxygen comments (``///``,
``//!``), tool pragmas (``// clang-format off``, ``// NOLINT``, ``# noqa`` and the
like). ``/* */`` comments are not read, and a ``//`` line inside one is not a
comment line. A license header (its notice through the closing ruler) is
skipped; lines just before or after it are checked as their own block. A ruler
line such as ``//-----`` stays in its block but does not count, so a banner is
one line.

    python .ci/scripts/check_comment_blocks.py --base origin/main --head HEAD

Exits 1 after an ``::error file=...`` line per block, 2 when git cannot diff
``--base``/``--head``.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAX_LINES = 2
SLASH_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".cl", ".i"}
HASH_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".toml", ".mk", ".cmake", ".bzl", ".bazel"}
HASH_NAMES = {"BUILD", "WORKSPACE", "CMakeLists.txt", "Dockerfile"}
_LICENSE = re.compile(r"(//|#)\s*(Copyright\b|SPDX-License-Identifier:|Licensed under the)")
_NOTICE_END = re.compile(r"(//|#)\s*limitations under the License\.$")
_PRAGMA = re.compile(
    r"(//|#)\s*(clang-format\b|NOLINT|NOSONAR|IWYU\b|noqa\b|type:|pragma:|fmt:|"
    r"nosec\b|isort:|shellcheck\s|yamllint\s|buildifier:|-\*-)"
)
_DOXYGEN = re.compile(r"//[/!]")
_RULER = re.compile(r"(//+|#+)\s*([-=*#~_+/])\2{2,}\s*$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def marker(path: str) -> str | None:
    """The line-comment marker the check reads in ``path``, or None to skip it."""
    name = path.rsplit("/", 1)[-1]
    suffix = Path(name).suffix
    if suffix in SLASH_SUFFIXES:
        return "//"
    if suffix in HASH_SUFFIXES or name in HASH_NAMES or name.lower().startswith("makefile"):
        return "#"
    return None


def _python_comments(text: str) -> set[int] | None:
    """Lines whose first token is a comment; None when ``text`` does not tokenize."""
    found = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT and not tok.line[: tok.start[1]].strip():
                found.add(tok.start[0])
    except (tokenize.TokenError, SyntaxError):
        return None
    return found


def _slash_comments(text: str) -> set[int]:
    """Lines that start with a ``//`` comment, outside any ``/* */`` comment or string literal."""
    found: set[int] = set()
    line, i, n = 1, 0, len(text)
    in_block, quote, line_start = False, "", True  # line_start: only blanks so far on this line
    while i < n:
        c, nxt = text[i], (text[i + 1] if i + 1 < n else "")
        if c == "\n":
            line, quote, line_start = line + 1, "", not in_block
            i += 1
        elif in_block:
            in_block = not (c == "*" and nxt == "/")
            line_start = False
            i += 1 if in_block else 2
        elif quote:
            if c == "\\" and nxt not in ("\n", ""):
                i += 2
            else:
                quote = "" if c == quote else quote
                i += 1
        elif c == "/" and nxt == "/":
            if line_start:
                found.add(line)
            end = text.find("\n", i)
            i, line_start = (n if end < 0 else end), False
        elif c == "/" and nxt == "*":
            in_block, line_start = True, False
            i += 2
        else:
            quote = c if c in "\"'" else quote
            line_start = line_start and c.isspace()
            i += 1
    return found


def _exempt(number: int, stripped: str) -> bool:
    return (
        (number == 1 and stripped.startswith("#!"))
        or bool(_DOXYGEN.match(stripped) and not _RULER.match(stripped))
        or bool(_PRAGMA.match(stripped))
    )


def comment_lines(text: str, path: str) -> set[int]:
    """1-based numbers of the full-line comments in ``text`` that count toward a block."""
    # Not splitlines(): it also splits on form feeds, so line numbers would drift from git's.
    lines, mark = text.split("\n"), marker(path)
    found = _python_comments(text) if path.endswith(".py") else None
    if found is None and mark == "//":
        found = _slash_comments(text)
    if found is None:
        found = {i for i, line in enumerate(lines, 1) if line.lstrip().startswith(mark)}
    return {i for i in found if not _exempt(i, lines[i - 1].strip())}


def blocks(numbers: set[int]) -> list[tuple[int, int]]:
    """(first, last) line of each run of consecutive numbers."""
    runs: list[tuple[int, int]] = []
    for n in sorted(numbers):
        if runs and n == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], n)
        else:
            runs.append((n, n))
    return runs


def _outside_license(lines: list[str], first: int, last: int) -> list[tuple[int, int]]:
    """The runs of the block ``first..last`` that are not its license header (notice and ruler)."""
    notice = [n for n in range(first, last + 1) if _LICENSE.match(lines[n - 1].strip())]
    if not notice:
        return [(first, last)]
    start = notice[0]
    end = next(
        (n for n in range(start, last + 1) if _NOTICE_END.match(lines[n - 1].strip())),
        notice[-1],
    )
    if end < last and _RULER.match(lines[end].strip()):
        end += 1
    return [run for run in ((first, start - 1), (end + 1, last)) if run[0] <= run[1]]


def long_blocks(text: str, path: str, changed: set[int]) -> list[tuple[int, int, int]]:
    """(first, last, length) of the blocks over ``MAX_LINES`` that hold a line of ``changed``."""
    lines, found = text.split("\n"), []
    for block_first, block_last in blocks(comment_lines(text, path)):
        for first, last in _outside_license(lines, block_first, block_last):
            body = [lines[n - 1].strip() for n in range(first, last + 1)]
            size = sum(not _RULER.match(line) for line in body)
            if size > MAX_LINES and any(first <= n <= last for n in changed):
                found.append((first, last, size))
    return found


def changed_paths(base: str, head: str) -> list[tuple[str, ...]]:
    """(path at head, path at base if renamed) of each file ``base...head`` adds or changes."""
    result = _git("diff", "-z", "-M", "--name-status", "--diff-filter=AMR", f"{base}...{head}")
    result.check_returncode()
    fields, paths, i = result.stdout.split("\0")[:-1], [], 0
    while i < len(fields):
        if fields[i].startswith("R"):
            paths.append((fields[i + 2], fields[i + 1]))
            i += 3
        else:
            paths.append((fields[i + 1],))
            i += 2
    return paths


def added_lines(base: str, head: str, paths: tuple[str, ...]) -> set[int]:
    """Line numbers at ``head`` that ``base...head`` adds or changes in ``paths[0]``."""
    result = _git("diff", "-U0", "-M", f"{base}...{head}", "--", *paths)
    result.check_returncode()
    lines: set[int] = set()
    for m in _HUNK.finditer(result.stdout):
        start = int(m.group(1))
        lines.update(range(start, start + int(m.group(2) or 1)))
    return lines


def findings(base: str, head: str) -> list[tuple[str, int, int, int]]:
    """(path, first, last, length) of every long block the PR adds to or edits."""
    found = []
    for paths in changed_paths(base, head):
        if marker(paths[0]) is None:
            continue
        shown = _git("show", f"{head}:{paths[0]}")
        shown.check_returncode()
        changed = added_lines(base, head, paths)
        found += [(paths[0], *block) for block in long_blocks(shown.stdout, paths[0], changed)]
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)
    try:
        found = findings(args.base, args.head)
    except subprocess.CalledProcessError as exc:
        print(f"cannot diff {args.base}...{args.head}: {exc.stderr.strip()}", file=sys.stderr)
        return 2
    for path, first, last, size in found:
        print(
            f"::error file={path},line={first},endLine={last},title=Comment block::"
            f"{size} comment lines; keep it to {MAX_LINES} that say the reason, "
            "and link an issue or PR for the rest (AGENTS.md)"
        )
    if found:
        return 1
    print(f"No comment block over {MAX_LINES} lines among the lines this PR adds or changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
