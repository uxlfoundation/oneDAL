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
  run.py validate                                  every task.json is well formed and its base and patches exist, no build
  run.py check   <task>...                         grader self-check, no LLM: untouched repo must fail, reference passes
  run.py agent   <task> <arm> <model> <rep> [--effort L]  prep a history-free repo for the arm, run `claude -p`, grade
  run.py matrix  [--tasks ..] [--arms ..] [--models ..] [--reps N] [-j N] [--batch NAME] [--effort L] [--force]
  run.py grade   <run_dir>...                      re-grade existing runs (deterministic; reruns the tests)
  run.py grade   --contract <run_dir> [--source DIR]   agent-benchmark entry point: grade.json as repo_grade.v1
  run.py oracle  --contract <run_dir> [--source DIR]   agent-benchmark entry point: apply the reference solution
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
import contract
import graders
from common import ARMS, bazel, meta, prep, sh, task_spec
import traces
from metrics import metrics

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


# oldest Claude Code measured to load AGENTS.md itself (only with no CLAUDE.md in its start-up chain); 2.1.241 and
# 2.1.250 never do, so on them main-raw is the same as none unless the agent opens the file
MIN_CLI = (2, 1, 286)


def run_agent(task, arm, model, rep, batch, effort=None):
    cli = config.claude_version()
    if not cli or tuple(int(x) for x in cli.split(".")[:3]) < MIN_CLI:
        raise SystemExit(f"{config.claude_bin()} is version {cli}; AGENTS.md needs Claude Code "
                         f"{'.'.join(map(str, MIN_CLI))}+ (set ONEDAL_EVAL_CLAUDE)")
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
    traces.convert(rd)  # events.jsonl, answer.txt, usage.json: what grading reads
    m = meta(rd)
    m.update(model=model, model_id=config.model_id(model), rep=rep, batch=batch, agent_rc=rc,
             effort=effort, cli_version=cli,
             wall_s=round(time.time() - t0))
    (rd / "meta.json").write_text(json.dumps(m))
    try:
        return grade(rd)
    finally:
        bazel(rd, rd / "repo", "shutdown", rd / "shutdown.log")


def run_inputs(rd):
    """(usage, trace) for the graders. A Claude trace.jsonl is converted first; nothing else reads it."""
    usage = traces.convert(rd) if (rd / "trace.jsonl").exists() else {}
    answer = rd / "answer.txt"
    m = meta(rd)
    return usage, {**metrics(rd), "answer": answer.read_text() if answer.exists() else "",
                   "start_sha": contract.start_commit(rd / "repo", task_spec(m["task"]), m)}


def grade(rd):
    m = meta(rd)
    t = task_spec(m["task"])
    usage, tr = run_inputs(rd)
    g = graders.GRADERS[t["grader"]](rd, rd / "repo", t, tr)
    row = {**m, **usage, **{k: v for k, v in tr.items() if k not in ("answer", "start_sha")}, **g}
    (rd / "grade.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: row.get(k) for k in HEADLINE if k in row}), flush=True)
    return row


def check(task, kind):
    """kind=null: untouched task repo (must fail). kind=oracle: reference fix/answer applied (must pass)."""
    t = task_spec(task)
    rd = config.root() / "check" / f"{task}__{kind}"
    repo = prep(task, "main-raw", rd)
    (rd / "answer.txt").write_text("")
    (rd / "events.jsonl").write_text("")
    if kind == "oracle":
        contract.oracle(rd, repo, t)
    try:
        return grade(rd)
    finally:
        bazel(rd, repo, "shutdown", rd / "shutdown.log")


def check_verdict(task):
    rows = {k: check(task, k) for k in ("null", "oracle")}
    t = task_spec(task)
    if t.get("family") == "review":
        # score-only tasks: the oracle must reach score.max on score.metric, as agent-benchmark's tasks check requires
        sc = t.get("score")
        reached = ((rows["oracle"].get(sc["metric"]) or 0) >= sc["max"]) if sc else rows["oracle"]["pass"] is True
        ok = reached and not rows["oracle"]["decoys_flagged"] and rows["null"].get("parse_error")
    else:
        ok = rows["null"]["pass"] is False and rows["oracle"]["pass"] is True
    print(f"{task}: {'OK' if ok else 'GRADER BROKEN'}", flush=True)
    return ok


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
        return r.returncode

    with ThreadPoolExecutor(a.j) as ex:
        rcs = list(ex.map(one, todo))
    # a child that crashed in prep, the agent call or grading wrote no grade.json: the cell is missing, not failed.
    # Report it in the exit status so a scripted matrix does not read an incomplete batch as a finished one.
    broken = [(j, rc) for j, rc in zip(todo, rcs) if rc]
    for j, rc in broken:
        print(f"run failed (rc={rc}): {'__'.join(map(str, j))}", file=sys.stderr)
    print(f"{len(todo) - len(broken)}/{len(todo)} runs graded", flush=True)
    return 1 if broken else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("validate")
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
    p.add_argument("--contract", action="store_true", help="agent-benchmark layout: workspace/, task.json, ...")
    p.add_argument("--source", type=Path, help="full-history oneDAL clone; overrides ONEDAL_EVAL_SRC")
    p = sub.add_parser("oracle")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--contract", action="store_true", required=True)
    p.add_argument("--source", type=Path, help="full-history oneDAL clone; overrides ONEDAL_EVAL_SRC")
    p = sub.add_parser("static")
    p.add_argument("--tree", type=Path, default=config.src())
    p.add_argument("--out", type=Path)
    p = sub.add_parser("summary")
    p.add_argument("--batch", default="default")
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
    elif a.cmd == "agent":
        run_agent(a.task, a.arm, a.model, a.rep, a.batch, a.effort)
    elif a.cmd == "matrix":
        sys.exit(matrix(a))
    elif a.cmd == "grade" and a.contract:
        for rd in a.run_dirs:
            rd = rd.resolve()
            try:
                g = contract.grade(rd)
                print(json.dumps({k: g[k] for k in ("task", "pass", "errors")}), flush=True)
            finally:
                bazel(rd, rd / "workspace", "shutdown", rd / "shutdown.log")
    elif a.cmd == "grade":
        for rd in a.run_dirs:
            rd = rd.resolve()
            try:
                grade(rd)
            finally:
                bazel(rd, rd / "repo", "shutdown", rd / "shutdown.log")
    elif a.cmd == "oracle":
        rd = a.run_dir.resolve()
        contract.oracle(rd, rd / "workspace", json.loads((rd / "task.json").read_text()))
    elif a.cmd == "static":
        import static_check
        static_check.main(a.tree.resolve(), a.out)
    elif a.cmd == "summary":
        import summarize
        summarize.main(config.root() / "runs" / a.batch)


if __name__ == "__main__":
    main()
