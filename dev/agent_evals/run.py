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

"""oneDAL agent-eval runner.

  run.py list                                      tasks and their grader
  run.py check   <task>...                         grader self-check, no LLM: untouched repo must fail, reference passes
  run.py agent   <task> <arm> <model> <rep> [--effort L]  prep a history-free repo for the arm, run `claude -p`, grade
  run.py matrix  [--tasks ..] [--arms ..] [--models ..] [--reps N] [-j N] [--batch NAME] [--effort L] [--force]
  run.py grade   <run_dir>...                      re-grade existing runs (deterministic; reruns the tests)
  run.py static  [--tree DIR] [--out FILE]         T0: check guidance claims against the tree, $0
  run.py summary [--batch NAME]                    per-cell table of every graded run in the batch

Configuration is by environment variable, see config.py. Runs go to $ONEDAL_EVAL_ROOT/runs/<batch>/.
"""
import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
import graders
from common import ARMS, bazel, meta, prep, sh, task_spec
from traces import parse_trace

HEADLINE = ("task", "arm", "model", "rep", "pass", "strict_pass", "recall", "decoys_flagged", "cost", "turns",
            "wall_s", "bg_denied", "guidance_read")


def all_tasks():
    return sorted(p.parent.name for p in config.TASKS.glob("*/task.json"))


def run_dir(batch, task, arm, model, rep):
    return config.root() / "runs" / batch / f"{task}__{arm}__{model}__r{rep}"


def write_settings(rd):
    """Per-run Claude config: nothing from the user's ~/.claude, plus the no-background hook."""
    cfg = rd / "claude-config"
    cfg.mkdir(parents=True, exist_ok=True)
    hook = {"type": "command", "command": f"{sys.executable} {config.BIN / 'no_background.py'}"}
    (cfg / "settings.json").write_text(json.dumps(
        {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [hook]}]}}, indent=1))


def kill_group(proc, grace=10):
    """SIGTERM the agent's whole process group (children left by a finished agent too), SIGKILL after grace."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        proc.poll()  # reap the leader, or its zombie keeps the group alive
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_agent(task, arm, model, rep, batch, effort=None):
    t = task_spec(task)
    rd = run_dir(batch, task, arm, model, rep)
    repo = prep(task, arm, rd)
    write_settings(rd)
    prompt = t["prompt"] + ("\n\nWork in the foreground: background commands are disabled here, and your final "
                            "answer ends the session.")
    cmd = [config.claude_bin(), "-p", prompt, "--model", config.model_id(model),
           "--output-format", "stream-json", "--verbose",
           "--permission-mode", "bypassPermissions", "--max-budget-usd", str(config.model_budget(model)),
           "--disallowedTools", "WebFetch", "WebSearch"] + (["--effort", effort] if effort else [])
    t0 = time.time()
    with open(rd / "trace.jsonl", "w") as out, open(rd / "agent.err", "w") as err:
        # own session, so the agent's foreground builds/tests can be killed with it before grading starts
        proc = subprocess.Popen(cmd, cwd=repo, env=config.run_env(rd), stdin=subprocess.DEVNULL, stdout=out,
                                stderr=err, start_new_session=True)
        try:
            rc = proc.wait(timeout=config.agent_timeout())
        except subprocess.TimeoutExpired:
            rc = "timeout"
        finally:
            kill_group(proc)
    m = meta(rd)
    m.update(model=model, model_id=config.model_id(model), rep=rep, batch=batch, agent_rc=rc,
             effort=effort, cli_version=config.claude_version(),
             wall_s=round(time.time() - t0))
    (rd / "meta.json").write_text(json.dumps(m))
    try:
        return grade(rd)
    finally:
        bazel(rd, "shutdown", rd / "shutdown.log")


def grade(rd):
    m = meta(rd)
    t = task_spec(m["task"])
    tr = parse_trace(rd)
    g = graders.GRADERS[t["grader"]](rd, t, tr)
    row = {**m, **{k: v for k, v in tr.items() if k != "result_text"}, **g}
    (rd / "grade.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: row.get(k) for k in HEADLINE if k in row}), flush=True)
    return row


def check(task, kind):
    """kind=null: untouched task repo (must fail). kind=oracle: reference fix/answer applied (must pass)."""
    t = task_spec(task)
    rd = config.root() / "check" / f"{task}__{kind}"
    repo = prep(task, "main-raw", rd)
    text = graders.ORACLES[t["grader"]](rd, repo, t) if kind == "oracle" else ""
    (rd / "trace.jsonl").write_text(json.dumps({"type": "result", "result": text}) + "\n")
    try:
        return grade(rd)
    finally:
        bazel(rd, "shutdown", rd / "shutdown.log")


def check_verdict(task):
    rows = {k: check(task, k) for k in ("null", "oracle")}
    t = task_spec(task)
    if t.get("family") == "review":
        ok = (rows["oracle"].get("recall") in (1.0, None) and not rows["oracle"]["decoys_flagged"]
              and rows["null"].get("parse_error"))
    else:
        ok = rows["null"]["pass"] is False and rows["oracle"]["pass"] is True
    print(f"{task}: {'OK' if ok else 'GRADER BROKEN'}", flush=True)
    return ok


def matrix(a):
    tasks = a.tasks.split(",") if a.tasks else all_tasks()
    jobs = [(t, arm, mdl, r) for t, arm, mdl, r in
            itertools.product(tasks, a.arms.split(","), a.models.split(","), range(1, a.reps + 1))]
    todo = [j for j in jobs if a.force or not (run_dir(a.batch, *j) / "grade.json").exists()]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already graded, running {len(todo)} with -j {a.j}", flush=True)

    def one(j):
        r = sh([sys.executable, str(Path(__file__).resolve()), "agent", *map(str, j), "--batch", a.batch,
                *(["--effort", a.effort] if a.effort else [])], check=False)
        (sys.stdout if r.returncode == 0 else sys.stderr).write(r.stdout + (r.stderr if r.returncode else ""))
        sys.stdout.flush()

    with ThreadPoolExecutor(a.j) as ex:
        list(ex.map(one, todo))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("check")
    p.add_argument("tasks", nargs="*")
    p = sub.add_parser("agent")
    for x in ("task", "arm", "model"):
        p.add_argument(x)
    p.add_argument("rep", type=int)
    p.add_argument("--batch", default="default")
    p.add_argument("--effort")
    p = sub.add_parser("matrix")
    p.add_argument("--tasks")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--models", default="haiku")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("-j", type=int, default=4)
    p.add_argument("--batch", default="default")
    p.add_argument("--effort", help="Claude Code --effort level, e.g. low")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("grade")
    p.add_argument("run_dirs", nargs="+", type=Path)
    p = sub.add_parser("static")
    p.add_argument("--tree", type=Path, default=config.src())
    p.add_argument("--out", type=Path)
    p = sub.add_parser("summary")
    p.add_argument("--batch", default="default")
    a = ap.parse_args()

    if a.cmd == "list":
        for t in all_tasks():
            s = task_spec(t)
            print(f"{t:28} {s.get('family', ''):8} {s['grader']}")
    elif a.cmd == "check":
        sys.exit(0 if all([check_verdict(t) for t in a.tasks or all_tasks()]) else 1)
    elif a.cmd == "agent":
        run_agent(a.task, a.arm, a.model, a.rep, a.batch, a.effort)
    elif a.cmd == "matrix":
        matrix(a)
    elif a.cmd == "grade":
        for rd in a.run_dirs:
            grade(rd.resolve())
    elif a.cmd == "static":
        import static_check
        static_check.main(a.tree.resolve(), a.out)
    elif a.cmd == "summary":
        import summarize
        summarize.main(config.root() / "runs" / a.batch)


if __name__ == "__main__":
    main()
