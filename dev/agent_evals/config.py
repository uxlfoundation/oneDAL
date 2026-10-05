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

"""Paths and environment for the oneDAL agent-eval tasks and graders.

Everything host-specific is read from environment variables so the graders run from any oneDAL checkout:

  ONEDAL_EVAL_SRC      oneDAL git clone with full history (default: the checkout this file lives in).
                       Mined tasks read fix commits from it, so it must contain them. `run.py grade|oracle
                       --source DIR` overrides it (agent-benchmark passes {source_dir}). Never visible to the agent.
  ONEDAL_EVAL_ROOT     where `run.py check` puts its run directories (default: $TMPDIR/onedal-agent-evals-$USER).
  ONEDAL_EVAL_BAZEL    real bazel/bazelisk binary (default: first bazelisk/bazel on PATH outside bin/).
  ONEDAL_EVAL_REPO_CACHE, ONEDAL_EVAL_DISK_CACHE   shared Bazel caches (default: under ONEDAL_EVAL_ROOT/cache).

Agent runs (models, harness, isolation) are agent-benchmark's: see README.md.
"""
import getpass
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

H = Path(__file__).resolve().parent
TASKS = H / "tasks"
BIN = H / "bin"
# Snapshots are built from a commit of SRC; this directory must never be visible to the agent.
EVAL_DIR_IN_REPO = "dev/agent_evals"


# set by `run.py grade|oracle --source`; takes precedence over ONEDAL_EVAL_SRC
SRC_OVERRIDE = None


def src():
    if SRC_OVERRIDE:
        return Path(SRC_OVERRIDE).resolve()
    if os.environ.get("ONEDAL_EVAL_SRC"):
        return Path(os.environ["ONEDAL_EVAL_SRC"]).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=H, text=True, capture_output=True)
    return Path(top.stdout.strip())


def _user():
    # containers run as an arbitrary uid (`docker run --user`) with no passwd entry and no $USER
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return str(os.getuid())


def root():
    return Path(os.environ.get("ONEDAL_EVAL_ROOT")
                or Path(tempfile.gettempdir()) / f"onedal-agent-evals-{_user()}").resolve()


def real_bazel():
    if os.environ.get("ONEDAL_EVAL_BAZEL"):
        return os.environ["ONEDAL_EVAL_BAZEL"]
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if Path(p).resolve() != BIN)
    for name in ("bazelisk", "bazel"):
        found = shutil.which(name, path=path)
        if found:
            return found
    raise SystemExit("no bazel/bazelisk on PATH; set ONEDAL_EVAL_BAZEL")


def run_env(rd):
    """Environment for graders: per-run Bazel output_base, shared caches, the bazel wrapper first on PATH."""
    r = root()
    e = dict(os.environ)
    e.pop("MKLROOT", None)  # an exported MKLROOT makes Bazel re-fetch @mkl as symlinks into it
    e.pop("ONEDAL_EVAL_SRC", None)  # full history holds every fix; graders read it in-process, never via bazel
    e["PATH"] = f"{BIN}{os.pathsep}{e['PATH']}"
    e["ONEDAL_EVAL_BAZEL"] = real_bazel()
    e["EVAL_BAZEL_OUTPUT_BASE"] = str(rd / "ob")
    e.setdefault("ONEDAL_EVAL_REPO_CACHE", str(r / "cache" / "repos"))
    e.setdefault("ONEDAL_EVAL_DISK_CACHE", str(r / "cache" / "disk"))
    return e
