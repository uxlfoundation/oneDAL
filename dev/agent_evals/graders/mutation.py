#===============================================================================
# Copyright Contributors to the oneDAL Project
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
#===============================================================================

"""Test-writing tasks graded by mutation testing: the agent's tests must pass and must fail on seeded bugs.

The agent's final answer ends with ```json {"targets": ["//<package>:<test target>", ...]}```.

task.json keys:
  package         package (or list of packages) the targets must live in, e.g. "cpp/oneapi/dal/algo/foo"
  mutants         patches relative to the task directory (default: mutants/*.patch). Each must compile and be
                  killed by the reference tests; a mutant that fails to build is recorded, not counted as a kill
  min_kill        pass threshold on kill_rate (default 0.6)
  oracle_targets  targets of the reference tests in tasks/<id>/oracle/<repo path> (BUILD files as BUILD.oracle)

Library sources (anything that is neither a test file nor a BUILD/.bzl file) are restored from base -- or
removed, if the agent added them -- before the targets run, so tests cannot pass by editing or adding the
code under test; the agent's versions are put back afterwards so that regrading gives the same result.
"""
import json
import re
import shutil
from pathlib import Path

import common
from common import bazel, changed_files, clang_format_ok, diff_lines, is_test_file, last_json, task_dir

TEST_MACRO = re.compile(r"^\s*(?:TEST|TEST_M|TEST_CASE\w*|TEMPLATE_\w*TEST\w*)\s*\(", re.M)
LABEL = re.compile(r"^//([A-Za-z0-9_/.-]+):([A-Za-z0-9_.-]+)$")


def packages(t):
    p = t["package"]
    return [p] if isinstance(p, str) else list(p)


def mutant_paths(t):
    d = task_dir(t["id"])
    return [d / m for m in t["mutants"]] if t.get("mutants") else sorted((d / "mutants").glob("*.patch"))


def is_build_file(f):
    return Path(f).name in ("BUILD", "BUILD.bazel") or f.endswith(".bzl")


def in_base(repo, base, f):
    return common.git(repo, "cat-file", "-e", f"{base}:{f}", check=False).returncode == 0


def authored_lines(ws, base, files):
    """Changed lines of tracked files plus every line of new (untracked) files."""
    repo = ws
    new = set(common.git(repo, "ls-files", "--others", "--exclude-standard", check=False).stdout.split())
    n = diff_lines(ws, base, [f for f in files if f not in new])
    return n + sum(len((repo / f).read_text(errors="replace").splitlines()) for f in files
                   if f in new and (repo / f).is_file())


def run_targets(rd, ws, targets, log):
    """bazel test; returns (rc, n_executed). rc 0 pass, 1 build failure, 3 test failure, 4 no tests."""
    rc, text = bazel(rd, ws, "test " + " ".join(targets) + " --nocache_test_results --test_output=errors", log)
    m = re.search(r"Executed (\d+) out of (\d+) test", text)
    return rc, int(m.group(1)) if m else 0


def g_mutation(rd, ws, t, tr):
    repo = ws
    base = tr["base_sha"]
    pkgs = packages(t)
    j = last_json(tr["answer"])
    targets = j.get("targets") if isinstance(j, dict) else None
    changed = changed_files(ws, base)
    # Library sources, whether the agent edited one that existed at base or added a new one: a task
    # that allows test and BUILD files only must not be passable by putting the behaviour under test
    # into a new non-test source, so both kinds are taken out of the tree before the targets run and
    # put back afterwards.
    src_modified = [f for f in changed if not is_test_file(f) and not is_build_file(f)]
    # the agent's work: files under the package that are not library sources, plus test files anywhere
    authored = [f for f in changed if f not in src_modified and (is_test_file(f) or any(
        f.startswith(p + "/") for p in pkgs))]
    out = {"pass": False, "targets": targets, "src_modified": src_modified, "authored_files": authored,
           "n_tests": sum(len(TEST_MACRO.findall((repo / f).read_text(errors="replace")))
                          for f in authored if f.endswith((".cpp", ".hpp")) and (repo / f).exists()),
           "diff_lines": authored_lines(ws, base, authored), "format_ok": clang_format_ok(repo, authored)}

    if not isinstance(targets, list) or not targets or not all(isinstance(x, str) for x in targets):
        return {**out, "why": "no targets reported"}
    bad = [x for x in targets if not (LABEL.match(x) and LABEL.match(x).group(1) in pkgs)]
    if bad:
        return {**out, "why": f"targets outside {pkgs} or malformed: {bad}"}
    if not authored:
        return {**out, "why": "no test or BUILD file added/changed in the package"}

    saved = {f: (repo / f).read_bytes() if (repo / f).exists() else None for f in src_modified}
    in_base_modified = [f for f in src_modified if in_base(repo, base, f)]
    if in_base_modified:
        common.git(repo, "checkout", base, "--", *in_base_modified)
    for f in src_modified:  # a library source the agent added has no base version to check out
        if f not in in_base_modified:
            (repo / f).unlink(missing_ok=True)
    try:
        rc, n_exec = run_targets(rd, ws, targets, rd / "grade.log")
        out.update(baseline_rc=rc, baseline_executed=n_exec, baseline_pass=rc == 0 and n_exec > 0)
        if not out["baseline_pass"]:
            return {**out, "why": "targets do not pass on the pristine library"}
        mutants = {}
        for p in mutant_paths(t):
            name = p.stem
            if common.git(repo, "apply", "--check", str(p), check=False).returncode:
                mutants[name] = {"apply_error": True, "killed": False, "build_error": False}
                continue
            common.git(repo, "apply", str(p))
            try:
                mrc, _ = run_targets(rd, ws, targets, rd / f"mutant_{name}.log")
            finally:
                common.git(repo, "apply", "-R", str(p))
            mutants[name] = {"rc": mrc, "killed": mrc == 3, "build_error": mrc not in (0, 3)}
    finally:
        for f, data in saved.items():  # put the agent's tree back: regrading must see the same inputs
            if data is None:
                (repo / f).unlink(missing_ok=True)
            else:
                (repo / f).write_bytes(data)
    live = [m for m in mutants.values() if not m["build_error"] and not m.get("apply_error")]
    kill_rate = round(sum(m["killed"] for m in live) / len(live), 3) if live else 0.0
    out.update(mutants=mutants, n_mutants=len(mutants), n_killed=sum(m["killed"] for m in live),
               n_build_error=sum(m["build_error"] for m in mutants.values()), kill_rate=kill_rate,
               survived=sorted(k for k, m in mutants.items() if not m["killed"] and not m["build_error"]))
    out["pass"] = kill_rate >= t.get("min_kill", 0.6)
    return out


def o_mutation(rd, repo, t):
    """Reference tests: tasks/<id>/oracle/<repo path>[.oracle] copied over the repo; answer names oracle_targets."""
    src = task_dir(t["id"]) / "oracle"
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            # stored as BUILD.oracle so the eval tree itself holds no Bazel packages
            dst = repo / (rel.with_suffix("") if rel.suffix == ".oracle" else rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
    return "```json\n" + json.dumps({"targets": t["oracle_targets"]}) + "\n```"


GRADERS = {"mutation": g_mutation}
ORACLES = {"mutation": o_mutation}
