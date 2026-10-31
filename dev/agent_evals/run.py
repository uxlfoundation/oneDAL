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

dokimi prepares workspaces, runs agents, self-checks the tasks (`dokimi repo tasks check`) and checks the guidance
(`dokimi repo scan`, README.md), and calls these entry points in the eval image. Configuration is by environment variable, see config.py.
"""
import argparse
import json
from pathlib import Path

import config
import contract
from common import bazel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
