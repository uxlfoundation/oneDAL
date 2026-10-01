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

"""Paths, models and environment for the oneDAL agent-eval harness.

Everything host-specific is read from environment variables so the harness runs from any oneDAL checkout:

  ONEDAL_EVAL_SRC      oneDAL git clone with full history (default: the checkout this file lives in).
                       Mined tasks read fix commits from it, so it must contain them.
  ONEDAL_EVAL_ROOT     where run directories go (default: $TMPDIR/onedal-agent-evals-$USER). Must have no
                       CLAUDE.md / AGENTS.md / .claude in any ancestor directory, or those files leak into every arm.
  ONEDAL_EVAL_BAZEL    real bazel/bazelisk binary (default: first bazelisk/bazel on PATH outside bin/).
  ONEDAL_EVAL_REPO_CACHE, ONEDAL_EVAL_DISK_CACHE   shared Bazel caches (default: under ONEDAL_EVAL_ROOT/cache).
  ONEDAL_EVAL_AGENT_TIMEOUT   seconds per agent run (default 2700).
  ONEDAL_EVAL_CLAUDE   Claude Code CLI binary (default: `claude` on PATH). The version matters: 2.1.277+ loads
                       AGENTS.md when no CLAUDE.md is present, older CLIs do not; each run records it.
"""
import getpass
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

H = Path(__file__).resolve().parent
TASKS = H / "tasks"
BIN = H / "bin"
# Snapshots are built from a commit of SRC; this directory must never be visible to the agent.
EVAL_DIR_IN_REPO = "dev/agent_evals"

MANIFEST = H / "repo-eval.yaml"


def manifest_guidance_globs(path=MANIFEST):
    """guidance.globs of the agent-benchmark manifest, the single list of guidance files the arms change.

    The harness is stdlib-only, so this reads just that line: `globs:` under `guidance:`, a one-line list of
    double-quoted strings (JSON syntax). Anything else is an error rather than a guess.
    """
    m = re.search(r"^guidance:\n(?:[ \t]+.*\n)*?[ \t]+globs:[ \t]*(\[.*\])[ \t]*$", path.read_text(), re.M)
    globs = json.loads(m.group(1)) if m else None
    if not (isinstance(globs, list) and globs and all(isinstance(g, str) for g in globs)):
        raise SystemExit(f"{path}: guidance.globs must be a one-line list of double-quoted strings")
    return globs


GUIDANCE_GLOBS = manifest_guidance_globs()
ANCESTOR_CONTEXT = ["CLAUDE.md", "CLAUDE.local.md", "AGENTS.md", ".claude"]

# Model aliases -> (Bedrock id, Anthropic API id), budget cap per run in USD.
MODELS = {
    "haiku": ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5-20251001", 2.0),
    "sonnet": ("us.anthropic.claude-sonnet-5", "claude-sonnet-5", 5.0),
    "opus": ("us.anthropic.claude-opus-5-5", "claude-opus-5-5", 10.0),
}


def model_id(alias):
    if alias not in MODELS:
        return alias  # full model id passed through
    bedrock, api, _ = MODELS[alias]
    return bedrock if os.environ.get("CLAUDE_CODE_USE_BEDROCK") else api


def model_budget(alias):
    return MODELS.get(alias, (None, None, 5.0))[2]


def src():
    if os.environ.get("ONEDAL_EVAL_SRC"):
        return Path(os.environ["ONEDAL_EVAL_SRC"]).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=H, text=True, capture_output=True)
    return Path(top.stdout.strip())


def root():
    r = Path(os.environ.get("ONEDAL_EVAL_ROOT")
             or Path(tempfile.gettempdir()) / f"onedal-agent-evals-{getpass.getuser()}").resolve()
    if not os.environ.get("ONEDAL_EVAL_ALLOW_ANCESTOR_CONTEXT"):
        for d in [r, *r.parents]:
            hits = [n for n in ANCESTOR_CONTEXT if (d / n).exists()]
            # ~/.claude is isolated by the per-run CLAUDE_CONFIG_DIR; any other hit is loaded as project memory
            if hits and not (hits == [".claude"] and d == Path.home()):
                raise SystemExit(f"ONEDAL_EVAL_ROOT={r}: {d} contains {hits}; Claude Code would load it into "
                                 "every run. Pick a root outside that tree.")
    return r


def real_bazel():
    if os.environ.get("ONEDAL_EVAL_BAZEL"):
        return os.environ["ONEDAL_EVAL_BAZEL"]
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep) if Path(p).resolve() != BIN)
    for name in ("bazelisk", "bazel"):
        found = shutil.which(name, path=path)
        if found:
            return found
    raise SystemExit("no bazel/bazelisk on PATH; set ONEDAL_EVAL_BAZEL")


def claude_bin():
    return os.environ.get("ONEDAL_EVAL_CLAUDE", "claude")


def claude_version():
    r = subprocess.run([claude_bin(), "--version"], capture_output=True, text=True)
    return r.stdout.split()[0] if r.returncode == 0 and r.stdout else None


def agent_timeout():
    return int(os.environ.get("ONEDAL_EVAL_AGENT_TIMEOUT", 45 * 60))


def run_env(rd):
    """Environment for the agent and for graders: per-run Bazel output_base and Claude config dir."""
    r = root()
    e = dict(os.environ)
    e.pop("MKLROOT", None)  # an exported MKLROOT makes Bazel re-fetch @mkl as symlinks into it
    e["PATH"] = f"{BIN}{os.pathsep}{e['PATH']}"
    e["ONEDAL_EVAL_BAZEL"] = real_bazel()
    e["EVAL_BAZEL_OUTPUT_BASE"] = str(rd / "ob")
    e["EVAL_RUN_DIR"] = str(rd)
    e.setdefault("ONEDAL_EVAL_REPO_CACHE", str(r / "cache" / "repos"))
    e.setdefault("ONEDAL_EVAL_DISK_CACHE", str(r / "cache" / "disk"))
    e["CLAUDE_CONFIG_DIR"] = str(rd / "claude-config")
    # headless runs end when the model ends its turn; a backgrounded build then looks like a failure
    e["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
    return e
