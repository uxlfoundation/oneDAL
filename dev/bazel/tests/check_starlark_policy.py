#!/usr/bin/env python3
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
"""Fails on Starlark constructs that break hermeticity, outside a reviewed allowlist.

The rules of dev/bazel/AGENTS.md, checked on the syntax tree rather than with a regex:
Starlark parses as Python, so a call is found whatever its formatting. An allowed site is
named by file and enclosing function, so moving code inside a function keeps it allowed
and copying it into another function does not.

Usage: python3 dev/bazel/tests/check_starlark_policy.py [repo_root]
"""

import ast
import pathlib
import sys

# (rule, file, enclosing function) -> why the exception is needed.
ALLOWED = {
    ("use_default_shell_env", "dev/bazel/release.bzl", "_copy"): (
        "cmd.exe needs SystemRoot; only set when the exec host is Windows"
    ),
    ("use_default_shell_env", "dev/bazel/cc.bzl", "_compile_windows_resource"): (
        "rc.exe/llvm-rc is resolved from PATH, not from the toolchain (dev/bazel/TODO.md)"
    ),
}

# Execution requirements that opt an action out of the sandbox or the caches.
_FORBIDDEN_TAGS = {"no-sandbox", "no-cache", "no-remote", "local"}

_SKIP_DIRS = {".git", "bazel-bin", "bazel-out", "bazel-testlogs"}


def _starlark_files(root):
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if any(part in _SKIP_DIRS or part.startswith("bazel-") for part in parts):
            continue
        # Repository-rule templates (`*.tpl.BUILD`) are not valid Starlark until substituted.
        if ".tpl." in path.name:
            continue
        if path.suffix == ".bzl" or path.name in ("BUILD", "BUILD.bazel"):
            yield path


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel):
        self.rel = rel
        self.functions = []
        self.violations = []

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def _report(self, rule, node, detail):
        function = self.functions[-1] if self.functions else "<module>"
        if (rule, self.rel, function) not in ALLOWED:
            self.violations.append(
                f"{self.rel}:{node.lineno}: {rule} in {function}(): {detail}"
            )

    def visit_keyword(self, node):
        if node.arg == "use_default_shell_env":
            if not (isinstance(node.value, ast.Constant) and node.value.value is False):
                self._report(
                    "use_default_shell_env",
                    node.value,
                    "pass the toolchain's env (cc_common.get_environment_variables) instead",
                )
        elif node.arg in ("execution_requirements", "tags"):
            for key in _string_constants(node.value):
                if key in _FORBIDDEN_TAGS:
                    self._report(key, node.value, "opts the action out of sandboxing or caching")
        self.generic_visit(node)


def _string_constants(node):
    if isinstance(node, ast.Dict):
        nodes = node.keys
    elif isinstance(node, (ast.List, ast.Tuple)):
        nodes = node.elts
    else:
        return []
    return [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def check(root):
    violations, errors = [], []
    for path in _starlark_files(root):
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as e:
            errors.append(f"{rel}:{e.lineno}: cannot parse: {e.msg}")
            continue
        visitor = _Visitor(rel)
        visitor.visit(tree)
        violations += visitor.violations
    stale = [
        f"{f}: allowlist entry {r} in {fn}() matches nothing; remove it"
        for r, f, fn in ALLOWED
        if not _site_exists(root, r, f, fn)
    ]
    return violations, errors, stale


def _site_exists(root, rule, rel, function):
    path = root / rel
    if not path.exists():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for kw in ast.walk(node):
                if isinstance(kw, ast.keyword) and kw.arg == rule:
                    return True
    return False


def main(argv):
    root = pathlib.Path(argv[1] if len(argv) > 1 else ".").resolve()
    violations, errors, stale = check(root)
    for line in errors + violations + stale:
        print(line)
    if errors or violations or stale:
        print(
            f"\n{len(violations)} violation(s), {len(stale)} stale allowlist entr(y/ies), "
            f"{len(errors)} parse error(s). See dev/bazel/AGENTS.md; a new exception needs an "
            "entry in ALLOWED with its reason."
        )
        return 1
    print(f"Starlark policy: OK ({len(ALLOWED)} allowed site(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
