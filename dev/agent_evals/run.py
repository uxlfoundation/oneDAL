#!/usr/bin/env python3
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

"""oneDAL agent-eval tasks: validation, grader self-check and the agent-benchmark entry points. No LLM.

  run.py list                                      tasks and their grader
  run.py validate                                  every task.json is well formed and its base and patches exist, no build
  run.py check   [<task>...]                       grader self-check: untouched workspace must fail, reference passes
  run.py grade   --contract <run_dir>... [--source DIR]   agent-benchmark entry point: grade.json as repo_grade.v1
  run.py oracle  --contract <run_dir> [--source DIR]      agent-benchmark entry point: apply the reference solution
  run.py static  [--tree DIR] [--out FILE]         T0: check guidance claims against the tree, $0

Agent runs are driven by agent-benchmark (README.md). Configuration is by environment variable, see config.py.
"""
import argparse
import json
import sys
from pathlib import Path

import config
import contract
import graders
from common import bazel, prep, sh, task_spec


def all_tasks():
    return sorted(p.parent.name for p in config.TASKS.glob("*/task.json"))


CORE_KEYS = ("id", "family", "base", "prompt", "grader")


def validate():
    """Cheap structural check of every task (no build, no LLM): what CI runs on every change to this directory."""
    errs = []
    for task in all_tasks():
        t, d = task_spec(task), config.TASKS / task
        errs += [f"{task}: missing key {k}" for k in CORE_KEYS if k not in t]
        if t.get("id") != task:
            errs.append(f"{task}: id {t.get('id')!r} differs from the directory name")
        base = t.get("base", "")
        if len(base.rstrip("^")) != 40:
            errs.append(f"{task}: base {base!r} is not a full 40-character SHA")
        if sh(["git", "cat-file", "-e", base + "^{commit}"], cwd=config.src(), check=False).returncode:
            errs.append(f"{task}: base {base!r} does not resolve in {config.src()}")
        if t.get("grader") not in graders.GRADERS:
            errs.append(f"{task}: unknown grader {t.get('grader')!r}")
        for f in [*t.get("setup_patches", []), t.get("setup_patch"), t.get("review_patch"), t.get("oracle_patch")]:
            if f and not (d / f).is_file():
                errs.append(f"{task}: {f} not found")
        for g in t.get("gate", []):
            if not (isinstance(g, str) or (isinstance(g, dict) and {"metric", "min"} <= g.keys())):
                errs.append(f"{task}: bad gate entry {g!r}")
    for e in errs:
        print(e, file=sys.stderr)
    print(f"{len(all_tasks())} tasks, {len(errs)} problems", flush=True)
    return 0 if not errs else 1


def check(task, kind):
    """kind=null: untouched workspace (must fail). kind=oracle: reference fix/answer applied (must pass)."""
    t = task_spec(task)
    rd = config.root() / "check" / f"{task}__{kind}"
    ws = prep(task, rd)
    if kind == "oracle":
        contract.oracle(rd, ws, t)
    try:
        g = contract.grade(rd)
    finally:
        bazel(rd, ws, "shutdown", rd / "shutdown.log")
    row = {"pass": g["pass"], **g["metrics"], **g["details"], "errors": g["errors"]}
    print(json.dumps({"task": task, "kind": kind, "pass": row["pass"], "errors": len(row["errors"])}), flush=True)
    return row


def check_verdict(task):
    rows = {k: check(task, k) for k in ("null", "oracle")}
    t = task_spec(task)
    if any(r["errors"] for r in rows.values()):
        ok = False  # a grader that crashes on the null or oracle workspace is a grader bug
    elif t.get("family") == "review":
        # score-only tasks: the oracle must reach score.max on score.metric, as agent-benchmark's tasks check requires
        sc = t.get("score")
        reached = ((rows["oracle"].get(sc["metric"]) or 0) >= sc["max"]) if sc else rows["oracle"]["pass"] is True
        ok = reached and not rows["oracle"].get("decoys_flagged") and rows["null"].get("parse_error")
    else:
        ok = rows["null"]["pass"] is False and rows["oracle"]["pass"] is True
    print(f"{task}: {'OK' if ok else 'GRADER BROKEN'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("validate")
    p = sub.add_parser("check")
    p.add_argument("tasks", nargs="*")
    p = sub.add_parser("grade")
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--contract", action="store_true", required=True,
                   help="agent-benchmark layout: workspace/, task.json, meta.json, events.jsonl, answer.txt")
    p.add_argument("--source", type=Path, help="full-history oneDAL clone; overrides ONEDAL_EVAL_SRC")
    p = sub.add_parser("oracle")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--contract", action="store_true", required=True)
    p.add_argument("--source", type=Path, help="full-history oneDAL clone; overrides ONEDAL_EVAL_SRC")
    p = sub.add_parser("static")
    p.add_argument("--tree", type=Path, default=config.src())
    p.add_argument("--out", type=Path)
    a = ap.parse_args()
    if getattr(a, "source", None):
        if not (a.source / ".git").exists() and not (a.source / "HEAD").is_file():
            raise SystemExit(f"--source {a.source}: not a git repository")
        config.SRC_OVERRIDE = a.source

    if a.cmd == "list":
        for t in all_tasks():
            s = task_spec(t)
            print(f"{t:28} {s.get('family', ''):8} {s['grader']}")
    elif a.cmd == "validate":
        sys.exit(validate())
    elif a.cmd == "check":
        sys.exit(0 if all([check_verdict(t) for t in a.tasks or all_tasks()]) else 1)
    elif a.cmd == "grade":
        for rd in a.run_dirs:
            rd = rd.resolve()
            try:
                g = contract.grade(rd)
                print(json.dumps({k: g[k] for k in ("task", "pass", "errors")}), flush=True)
            finally:
                bazel(rd, rd / "workspace", "shutdown", rd / "shutdown.log")
    elif a.cmd == "oracle":
        rd = a.run_dir.resolve()
        contract.oracle(rd, rd / "workspace", json.loads((rd / "task.json").read_text()))
    elif a.cmd == "static":
        import static_check
        static_check.main(a.tree.resolve(), a.out)


if __name__ == "__main__":
    main()
