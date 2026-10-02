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
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import config


def sh(argv, cwd=None, env=None, check=True, timeout=None):
    """Run an argv list (never through a shell)."""
    r = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{argv}: rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
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
    archive = subprocess.Popen(["git", "-C", str(config.src()), "archive", commit], stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout)
    archive.stdout.close()
    if archive.wait() or tar.returncode:
        raise RuntimeError(f"git archive {commit} | tar: rc={archive.returncode}/{tar.returncode}")
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
    # guidance present as-is. Claude Code 2.1.277+ loads AGENTS.md when there is no CLAUDE.md (nested ones once the
    # agent touches that directory); run.py refuses older CLIs, which load it only if the agent opens the file
    pass


# New arms (other guidance revisions, placebo text, tooling) are one function each: mutate the task repo in place.
ARMS = {"none": arm_none, "main-raw": arm_main_raw}


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
        sh(["git", "-c", "user.name=Dev Contributor", "-c", "user.email=dev@example.com", "am", "-q",
            str(task_dir(task) / t["review_patch"])], cwd=repo)
    # base_* as agent-benchmark writes them (the upstream commit); start_sha is the workspace commit graders diff from
    up = sh(["git", "log", "-1", "--format=%H %cI", f"{t['base']}^{{commit}}"], cwd=config.src()).stdout.split()
    start = git(repo, "rev-parse", "HEAD").stdout.strip()
    (rd / "meta.json").write_text(json.dumps({"task": task, "arm": arm, "base_rev": t["base"], "base_sha": up[0],
                                              "base_date": up[1], "start_sha": start}))
    return repo


# ---------------------------------------------------------------- grading helpers
def bazel(rd, ws, args, log):
    """Run bazel in workspace ws with the run's output_base (rd/ob); returns (rc, log text).

    args is split with shlex and executed without a shell, so it cannot chain or redirect commands.
    """
    env = config.run_env(rd)
    with open(log, "w") as out:
        rc = subprocess.run([str(config.BIN / "bazel"), *shlex.split(args)], cwd=ws, env=env,
                            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT).returncode
    return rc, Path(log).read_text(errors="replace")


def last_json(text):
    """Last parseable ```json fenced block of the agent's final answer."""
    for b in reversed(re.findall(r"```json\s*(.*?)```", text or "", re.S)):
        try:
            return json.loads(b)
        except ValueError:
            continue
    return None


def changed_files(ws, base):
    """Files the agent changed or added in workspace ws since the prepared commit base."""
    ch = git(ws, "diff", "--name-only", base, check=False).stdout.split()
    ch += git(ws, "ls-files", "--others", "--exclude-standard", check=False).stdout.split()
    return sorted(set(ch) - {""})


def is_test_file(f):
    return "/test/" in f or f.endswith("_test.cpp")


@contextlib.contextmanager
def files_restored(repo, paths):
    """Put `paths` back as they were on exit, so grading leaves the run as the agent left it.

    Graders write hidden tests and revert library sources over the agent's tree. Without this,
    the first grade becomes part of the run and a regrade reads the grader's own files as the
    agent's work (`changed_files` is computed from the tree, not from a record).
    """
    saved = {f: ((repo / f).read_bytes() if (repo / f).is_file() else None) for f in paths}
    try:
        yield
    finally:
        for f, data in saved.items():
            p = repo / f
            if data is None:
                p.unlink(missing_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)


def diff_lines(ws, base, files):
    if not files:
        return 0
    ds = git(ws, "diff", "--shortstat", base, "--", *files, check=False).stdout
    return sum(int(x) for x in re.findall(r"(\d+) (?:insertion|deletion)", ds))


def clang_format_ok(repo, files):
    """clang-format 20.1.8 (the version CI pins) on changed C/C++ files; None if nothing to check.

    ONEDAL_EVAL_CLANG_FORMAT names an installed clang-format of that version (the eval image sets it); else uvx.
    """
    cxx = [f for f in files if re.search(r"\.(c|cpp|h|hpp|i|cl)$", f) and (repo / f).exists()]
    if not cxx:
        return None
    exe = os.environ.get("ONEDAL_EVAL_CLANG_FORMAT")
    tool = [exe] if exe else ["uvx", "--from", "clang-format==20.1.8", "clang-format"]
    r = sh([*tool, "--dry-run", "-Werror", "--style=file", *cxx], cwd=repo, check=False)
    return r.returncode == 0
