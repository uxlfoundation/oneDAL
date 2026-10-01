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

"""Entry points of the agent-benchmark repository-evaluation contract (see repo-eval.yaml).

  run.py grade --contract <run_dir>    reads workspace/, task.json, answer.txt, events.jsonl, meta.json from
                                       run_dir and writes run_dir/grade.json as repo_grade.v1
  run.py oracle --contract <run_dir>   applies the task's reference solution to run_dir/workspace and, for tasks
                                       graded on the final answer, writes the reference answer to answer.txt

The graders are the ones `run.py grade` uses; only the inputs and the output shape differ.
"""
import json
import traceback

import graders
from common import git
from metrics import metrics


def prepared_commit(ws, t, m):
    """The commit the agent started from: meta.json base_sha if given, else derived from the workspace history.

    Workspace preparation makes one root commit (snapshot + setup patches + arm) and, for review tasks, one
    `git am` commit on top; whatever follows is the agent's.
    """
    if m.get("base_sha"):
        return m["base_sha"]
    revs = git(ws, "rev-list", "--reverse", "--first-parent", "HEAD").stdout.split()
    return revs[1 if t.get("review_patch") else 0]


def split_result(g):
    """repo_grade.v1: numbers, booleans and null under metrics; strings, lists and objects under details."""
    mets, det = {}, {}
    for k, v in g.items():
        if k == "pass":
            continue
        if v is None or isinstance(v, (bool, int, float)):
            mets[k] = v
        else:
            det[k] = v
    return mets, det


def oracle(rd, ws, t):
    """Reference solution in place; also what `run.py check` uses for its oracle half."""
    text = graders.ORACLES[t["grader"]](rd, ws, t)
    if text:
        (rd / "answer.txt").write_text(text)
    return text


def grade(rd):
    t = json.loads((rd / "task.json").read_text())
    out = {"schema": "repo_grade.v1", "task": t.get("id"), "grader": t.get("grader"), "pass": None,
           "metrics": {}, "details": {}, "errors": []}
    try:
        ws = rd / "workspace"
        m = json.loads((rd / "meta.json").read_text()) if (rd / "meta.json").exists() else {}
        answer = rd / "answer.txt"
        tr = {**metrics(rd), "answer": answer.read_text() if answer.exists() else "",
              "base_sha": prepared_commit(ws, t, m)}
        g = graders.GRADERS[t["grader"]](rd, ws, t, tr)
        process = {k: v for k, v in tr.items() if k not in ("answer", "base_sha")}
        out["metrics"], out["details"] = split_result({**process, **g})
        for k in t.get("gate", []):  # a gate metric the grader did not reach is unmeasured, not failed
            out["metrics"].setdefault(k, None)
        out["pass"] = g.get("pass")
    except Exception as e:  # report as not graded, never as a failure
        out.update(metrics={}, details={})
        out["pass"] = None
        out["errors"] = [f"{type(e).__name__}: {e}", traceback.format_exc(limit=5)]
    (rd / "grade.json").write_text(json.dumps(out, indent=1))
    return out
