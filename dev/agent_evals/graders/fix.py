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

"""Find-and-fix tasks: hidden tests restored over the agent's tree, then regression suites and hygiene."""
import common
from common import (bazel, changed_files, clang_format_ok, diff_lines, files_restored, is_test_file, show,
                    task_dir)


def g_test_pass(rd, ws, t, tr):
    """pass = hidden targets pass. Also: regress_pass, format_ok, diff_lines, root_cause_fixed.

    Strict pass (what to report) = pass and regress_pass.
    """
    repo = ws
    # `_changed_files`: the file list as it was before the caller wrote anything of its own
    # (graders/fix_hidden.py restores hidden files first and then delegates here, so reading
    # the tree at this point would report the grader's files as the agent's changes).
    # Tested for presence, not truthiness: an agent that changed nothing passes [] here.
    changed = t["_changed_files"] if "_changed_files" in t else changed_files(ws, tr["start_sha"])
    touched_tests = [f for f in changed if is_test_file(f)]
    src_changed = [f for f in changed if f not in touched_tests]
    # The hidden tests are written over the agent's tree only for the duration of grading:
    # left behind, they would be read back as the agent's own files on a regrade.
    with files_restored(repo, t["hidden_files"]):
        for f in t["hidden_files"]:  # restore hidden/pristine tests over whatever the agent did to them
            (repo / f).parent.mkdir(parents=True, exist_ok=True)
            (repo / f).write_text(show(t["hidden_from"], f))
        rc, _ = bazel(rd, ws, "test " + " ".join(t["targets"]) + " --test_output=errors --nocache_test_results",
                      rd / "grade.log")
        out = {"pass": rc == 0, "grade_rc": rc, "changed_files": src_changed, "touched_tests": touched_tests}
        if t.get("regression_targets"):
            # Same flag as the hidden run: the agent shares this run's Bazel output base, so without
            # it a result the agent itself cached would be accepted instead of an executed suite.
            rrc, _ = bazel(rd, ws, "test " + " ".join(t["regression_targets"])
                           + " --test_output=errors --nocache_test_results", rd / "regress.log")
            out["regress_pass"] = rrc == 0
            out["strict_pass"] = out["pass"] and out["regress_pass"]
        out["format_ok"] = clang_format_ok(repo, src_changed)
        out["diff_lines"] = diff_lines(ws, tr["start_sha"], src_changed)
        if t.get("root_cause"):
            rcf = t["root_cause"]
            out["root_cause_fixed"] = rcf["must_contain"] in (repo / rcf["file"]).read_text()
    return out


def o_test_pass(rd, repo, t):
    """Reference fix: oracle_patch, else reverse the setup patches, else the mined fix commit's non-test files."""
    if t.get("oracle_patch"):
        common.git(repo, "apply", str(task_dir(t["id"]) / t["oracle_patch"]))
    elif t.get("setup_patch") or t.get("setup_patches"):
        for p in reversed(t.get("setup_patches", [t.get("setup_patch")])):
            common.git(repo, "apply", "-R", str(task_dir(t["id"]) / p))
    else:
        fix = t["base"].rstrip("^")
        for f in common.sh(["git", "show", "--name-only", "--format=", fix], cwd=common.config.src()).stdout.split():
            if f not in t["hidden_files"]:
                (repo / f).parent.mkdir(parents=True, exist_ok=True)
                (repo / f).write_text(show(fix, f))
    return ""


GRADERS = {"test_pass": g_test_pass}
ORACLES = {"test_pass": o_test_pass}
