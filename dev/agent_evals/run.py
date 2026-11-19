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

"""oneDAL agent-eval entry points for dokimi (repo-eval.yaml). No LLM.

  run.py grade   --contract <run_dir>... [--source DIR]   grade.json as repo_grade.v1
  run.py oracle  --contract <run_dir> [--source DIR]      apply the reference solution
  run.py list    [--tier T ...] [--all] [--smoke]         task ids by tier (see tasks/README.md); --smoke: the CI set
  run.py validate                                         every task.json has a known tier
  run.py bazel-versions [--check]                         Bazel versions the task bases pin; --check: vs task-bazel-versions.txt

dokimi prepares workspaces, runs agents, self-checks the tasks (`dokimi repo tasks check`) and checks the guidance
(`dokimi repo scan`, README.md), and calls these entry points in the eval image. Configuration is by environment variable, see config.py.
"""
import argparse
import json
import subprocess
from pathlib import Path

import config
import contract
from common import bazel

# task.json "tier": smoke (environment floor, scored), scored (default when absent), hard (ceiling probe, reported
# apart from the headline), retired (out of CI's self-check and the headline; dokimi repo run still takes them)
TIERS = ("smoke", "scored", "hard", "retired")
BAZEL_VERSIONS = config.H / "task-bazel-versions.txt"


def load_tasks():
    """One dict per task directory (id, tier, grader, base); exits on a task.json that cannot be read or has an unknown tier."""
    out, bad = [], []
    for d in sorted(config.TASKS.iterdir()):
        f = d / "task.json"
        if not f.is_file():
            continue
        try:
            t = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            bad.append(f"{f}: not valid JSON: {e}")
            continue
        tier = t.get("tier", "scored")
        if tier not in TIERS:
            bad.append(f"{f}: tier {tier!r} is not one of {', '.join(TIERS)}")
        out.append({"id": d.name, "tier": tier, "grader": t.get("grader", ""), "base": t.get("base", "")})
    if bad:
        raise SystemExit("\n".join(bad))
    return out


def task_tiers():
    """(id, tier) for every task directory."""
    return [(t["id"], t["tier"]) for t in load_tasks()]


def smoke_tasks():
    """The CI self-check set for a change outside the task directories: every smoke-tier task, plus one task for each
    grader no smoke task covers (the non-retired task of that grader with the best tier, then id order)."""
    rows = load_tasks()
    chosen = [t["id"] for t in rows if t["tier"] == "smoke"]
    covered = {t["grader"] for t in rows if t["tier"] == "smoke"}
    rank = {"smoke": 0, "scored": 1, "hard": 2}
    for g in sorted({t["grader"] for t in rows if t["tier"] != "retired"} - covered):
        cands = sorted((rank[t["tier"]], t["id"]) for t in rows if t["grader"] == g and t["tier"] != "retired")
        chosen.append(cands[0][1])
    return sorted(set(chosen))


def base_bazel_versions():
    """Sorted distinct Bazel versions: each task base's .bazelversion at the base commit (full history needed)."""
    found = set()
    for t in load_tasks():
        if not t["base"]:
            continue
        r = subprocess.run(["git", "-C", str(config.src()), "show", f"{t['base']}:.bazelversion"],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"{t['id']}: base {t['base']} has no readable .bazelversion: {r.stderr.strip()}")
        found.add(r.stdout.strip())
    return sorted(found, key=lambda v: tuple(int(x) for x in v.split(".")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--tier", action="append", choices=TIERS, help="only this tier; repeatable")
    p.add_argument("--all", action="store_true", help="with no --tier, include retired tasks too")
    p.add_argument("--smoke", action="store_true", help="the CI self-check set (smoke_tasks)")
    sub.add_parser("validate")
    p = sub.add_parser("bazel-versions")
    p.add_argument("--check", action="store_true", help=f"fail unless {BAZEL_VERSIONS.name} lists exactly these")
    p = sub.add_parser("grade")
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--contract", action="store_true", required=True,
                   help="dokimi run directory layout: workspace/, task.json, meta.json, events.jsonl, answer.txt")
    p.add_argument("--source", type=Path, help="full-history oneDAL clone; overrides ONEDAL_EVAL_SRC")
    p = sub.add_parser("oracle")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--contract", action="store_true", required=True)
    p.add_argument("--source", type=Path, help="full-history oneDAL clone; overrides ONEDAL_EVAL_SRC")
    a = ap.parse_args()
    if a.cmd == "list":
        if a.smoke:
            print("\n".join(smoke_tasks()))
            return
        for name, tier in task_tiers():
            wanted = tier in a.tier if a.tier else (a.all or tier != "retired")
            if not wanted:
                continue
            print(name)
        return
    if a.cmd == "validate":
        print(f"{len(task_tiers())} tasks, tiers ok")
        return
    if a.cmd == "bazel-versions":
        versions = base_bazel_versions()
        if not a.check:
            print("\n".join(versions))
            return
        listed = [v for v in BAZEL_VERSIONS.read_text().split() if v] if BAZEL_VERSIONS.is_file() else []
        if listed != versions:
            raise SystemExit(f"{BAZEL_VERSIONS.name} lists {' '.join(listed) or 'nothing'}; the task bases pin "
                             f"{' '.join(versions)}. Regenerate: run.py bazel-versions > {BAZEL_VERSIONS}")
        print(f"{BAZEL_VERSIONS.name}: {' '.join(versions)}")
        return
    if getattr(a, "source", None):
        if not (a.source / ".git").exists() and not (a.source / "HEAD").is_file():
            raise SystemExit(f"--source {a.source}: not a git repository")
        config.SRC_OVERRIDE = a.source

    if a.cmd == "grade":
        for rd in a.run_dirs:
            rd = rd.resolve()
            try:
                g = contract.grade(rd)
                print(json.dumps({k: g[k] for k in ("task", "pass", "errors")}), flush=True)
            finally:
                bazel(rd, rd / "workspace", "shutdown", rd / "shutdown.log")
    elif a.cmd == "oracle":
        rd = a.run_dir.resolve()
        try:
            contract.oracle(rd, rd / "workspace", json.loads((rd / "task.json").read_text()))
        finally:  # the reference build (release_path) starts a Bazel server in this run's output base
            bazel(rd, rd / "workspace", "shutdown", rd / "shutdown.log")


if __name__ == "__main__":
    main()
