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

"""Fix tasks whose hidden tests are synthetic (not in any commit) and/or need extra Bazel flags.

Grader `test_pass_files`: same result keys as `test_pass` (graders/fix.py), which it delegates to. Extra task keys:

  hidden_files        repo paths restored over the agent's tree before grading. Each is taken from
                      tasks/<id>/hidden/<path> (or <path>.oracle, used for BUILD files) when that file
                      exists, else from the `hidden_from` commit
                      (so mined and synthetic hidden files can be mixed).
  test_flags          extra bazel flags for the hidden `targets` run, e.g. ["--cpu=all"]
  regression_flags    extra bazel flags for the `regression_targets` run
  root_cause          {"file", "must_contain"} and/or {"file", "must_not_contain"}; root_cause_fixed is True
                      only if every given condition holds (a missing file counts as not fixed)

touched_tests reports the test files the agent changed, not the hidden files the grader wrote.
"""
from common import changed_files, is_test_file, show, task_dir
from graders.fix import g_test_pass, o_test_pass


def g_test_pass_files(rd, t, tr):
    repo = rd / "repo"
    agent_touched = [f for f in changed_files(rd) if is_test_file(f)]
    for f in t["hidden_files"]:
        src = task_dir(t["id"]) / "hidden" / f
        if not src.is_file():
            # BUILD files are stored as BUILD.oracle so the eval tree itself holds no Bazel packages
            src = src.with_name(src.name + ".oracle")
        text = src.read_text() if src.is_file() else show(t["hidden_from"], f)
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text(text)
    t2 = {k: v for k, v in t.items() if k != "root_cause"}
    t2["hidden_files"] = []  # already restored above
    t2["targets"] = list(t["targets"]) + list(t.get("test_flags", []))
    if t.get("regression_targets"):
        t2["regression_targets"] = list(t["regression_targets"]) + list(t.get("regression_flags", []))
    out = g_test_pass(rd, t2, tr)
    out["touched_tests"] = agent_touched
    if t.get("root_cause"):
        rcf = t["root_cause"]
        p = repo / rcf["file"]
        text = p.read_text() if p.is_file() else None
        ok = text is not None
        if ok and "must_contain" in rcf:
            ok = rcf["must_contain"] in text
        if ok and "must_not_contain" in rcf:
            ok = rcf["must_not_contain"] not in text
        out["root_cause_fixed"] = ok
    return out


GRADERS = {"test_pass_files": g_test_pass_files}
ORACLES = {"test_pass_files": o_test_pass}
