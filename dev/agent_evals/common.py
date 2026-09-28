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

"""Shell, git, snapshot and arm helpers shared by the runner and the graders."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import config


def sh(cmd, cwd=None, env=None, check=True, timeout=None):
    r = subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), text=True,
                       capture_output=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}: rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def git(repo, *a, **kw):
    return sh(["git", "-c", "user.name=eval", "-c", "user.email=eval@example.com", *a], cwd=repo, **kw)


def task_dir(task):
    return config.TASKS / task


def task_spec(task):
    t = json.loads((task_dir(task) / "task.json").read_text())
    t.setdefault("id", task)
    return t


def meta(rd):
    return json.loads((rd / "meta.json").read_text())


def show(rev, path):
    """File content at rev in the source clone (hidden tests, mined fixes)."""
    return sh(["git", "show", f"{rev}:{path}"], cwd=config.src()).stdout


def snapshot(commit, dest):
    """History-free tree of oneDAL at commit: no git log to leak the fix, no eval tasks to leak the answers."""
    dest.mkdir(parents=True)
    sh(f"git -C {config.src()} archive {commit} | tar -x -C {dest}")
    shutil.rmtree(dest / config.EVAL_DIR_IN_REPO, ignore_errors=True)
    git(dest, "init", "-q")
    git(dest, "add", "-A")
    git(dest, "commit", "-qm", f"snapshot {commit}")


# ---------------------------------------------------------------- arms
def guidance_files(repo):
    return sorted({p for g in config.GUIDANCE_GLOBS for p in repo.glob(g)})


def arm_none(repo):
    for p in guidance_files(repo):
        p.unlink()


def arm_main_raw(repo):
    pass  # guidance present as-is; Claude Code only auto-loads CLAUDE.md, so it is read only if the agent opens it


def arm_main_claude(repo):
    # CLAUDE.md shim next to every AGENTS.md, so Claude loads the hierarchy the way Codex loads AGENTS.md
    for p in guidance_files(repo):
        if p.name == "AGENTS.md":
            (p.parent / "CLAUDE.md").write_text("@AGENTS.md\n")


# New arms (other guidance revisions, placebo text, tooling) are one function each: mutate the task repo in place.
ARMS = {"none": arm_none, "main-raw": arm_main_raw, "main-claude": arm_main_claude}


def prep(task, arm, rd):
    t = task_spec(task)
    repo = rd / "repo"
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)
    snapshot(t["base"], repo)
    for p in t.get("setup_patches", [t["setup_patch"]] if t.get("setup_patch") else []):
        git(repo, "apply", str(task_dir(task) / p))
    ARMS[arm](repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--amend", "-m", "snapshot")
    if t.get("review_patch"):
        sh(f"git -c user.name='Dev Contributor' -c user.email=dev@example.com am -q "
           f"{task_dir(task) / t['review_patch']}", cwd=repo)
    base = git(repo, "rev-parse", "HEAD").stdout.strip()
    (rd / "meta.json").write_text(json.dumps({"task": task, "arm": arm, "base_sha": base}))
    return repo


# ---------------------------------------------------------------- grading helpers
def bazel(rd, args, log):
    """Run bazel in the run's repo with the run's output_base; returns (rc, log text)."""
    r = sh(["bash", "-c", f"bazel {args} > {log} 2>&1; echo $?"], cwd=rd / "repo", env=config.run_env(rd),
           check=False)
    return int(r.stdout.strip().splitlines()[-1] or 1), (Path(log).read_text() if Path(log).exists() else "")


def last_json(text):
    """Last parseable ```json fenced block of the agent's final answer."""
    for b in reversed(re.findall(r"```json\s*(.*?)```", text or "", re.S)):
        try:
            return json.loads(b)
        except ValueError:
            continue
    return None


def changed_files(rd):
    repo = rd / "repo"
    base = meta(rd)["base_sha"]
    ch = git(repo, "diff", "--name-only", base, check=False).stdout.split()
    ch += git(repo, "ls-files", "--others", "--exclude-standard", check=False).stdout.split()
    return sorted(set(ch) - {""})


def is_test_file(f):
    return "/test/" in f or f.endswith("_test.cpp")


def diff_lines(rd, files):
    if not files:
        return 0
    ds = git(rd / "repo", "diff", "--shortstat", meta(rd)["base_sha"], "--", *files, check=False).stdout
    return sum(int(x) for x in re.findall(r"(\d+) (?:insertion|deletion)", ds))


def clang_format_ok(repo, files):
    """clang-format 20.1.8 (the version CI pins) on changed C/C++ files; None if nothing to check."""
    cxx = [f for f in files if re.search(r"\.(c|cpp|h|hpp|i|cl)$", f) and (repo / f).exists()]
    if not cxx:
        return None
    r = sh(["uvx", "--from", "clang-format==20.1.8", "clang-format", "--dry-run", "-Werror", "--style=file", *cxx],
           cwd=repo, check=False)
    return r.returncode == 0
