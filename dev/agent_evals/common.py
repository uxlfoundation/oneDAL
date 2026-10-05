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


def prep(task, rd):
    """Workspace for the self-check, laid out as agent-benchmark prepares one (contract §3, guidance as checked in).

    rd/workspace is a history-free tree at the task base with the setup patches committed and, for review tasks,
    the change under review as its HEAD commit; rd also gets task.json, meta.json and an empty answer and trace.
    """
    t = task_spec(task)
    ws = rd / "workspace"
    if rd.exists():
        shutil.rmtree(rd)
    rd.mkdir(parents=True)
    snapshot(t["base"], ws)
    for p in t.get("setup_patches", [t["setup_patch"]] if t.get("setup_patch") else []):
        git(ws, "apply", str(task_dir(task) / p))
    git(ws, "add", "-A")
    git(ws, "commit", "-q", "--amend", "-m", "snapshot")
    if t.get("review_patch"):
        sh(["git", "-c", "user.name=Dev Contributor", "-c", "user.email=dev@example.com", "am", "-q",
            str(task_dir(task) / t["review_patch"])], cwd=ws)
    # base_* as agent-benchmark writes them (the upstream commit); start_sha is the workspace commit graders diff from
    up = sh(["git", "log", "-1", "--format=%H %cI", f"{t['base']}^{{commit}}"], cwd=config.src()).stdout.split()
    start = git(ws, "rev-parse", "HEAD").stdout.strip()
    (rd / "meta.json").write_text(json.dumps({"task": task, "arm": "guidance:raw", "base_rev": t["base"],
                                              "base_sha": up[0], "base_date": up[1], "start_sha": start}))
    (rd / "task.json").write_text(json.dumps(t, indent=1))
    (rd / "answer.txt").write_text("")
    (rd / "events.jsonl").write_text("")
    return ws


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


# agent-benchmark mounts the run directory here in every container (and the workspace at /eval/run/workspace)
CONTRACT_RUN_DIR = Path("/eval/run")


def rebase_run_path(p, rd, ws):
    """Map an absolute path recorded in a run to the run directory rd it is graded in now, or None.

    A run's answer and its symlinks (Bazel's convenience links point into the output base) name absolute paths
    from where the run was recorded: the standalone run directory, or /eval/run under agent-benchmark. A copied
    or archived run is graded somewhere else, so the prefix up to the run directory is replaced by rd, and its
    `repo`/`workspace` child by ws. A path under no run directory maps to None.
    """
    p = Path(p)
    if not p.is_absolute():
        return ws / p
    rest = None
    for root in (rd, rd.resolve(), CONTRACT_RUN_DIR):
        if p.is_relative_to(root):
            rest = p.relative_to(root).parts
            break
    else:
        # recorded under a run directory of the same name elsewhere (the original of a copied run)
        hits = [i for i, x in enumerate(p.parts) if x == rd.name]
        if hits:
            rest = p.parts[hits[-1] + 1:]
    if rest is None:
        return None
    if rest and rest[0] in ("repo", "workspace"):
        return ws.joinpath(*rest[1:])
    return rd.joinpath(*rest)


def resolve_in_run(p, rd, ws, max_links=40):
    """Resolve path p like Path.resolve, but inside run directory rd only; None if it leaves rd or is missing.

    Absolute paths, including absolute symlink targets met on the way, go through rebase_run_path, so a
    run graded away from where it was recorded resolves to its own files and never to the original's.
    """
    top = rd.resolve()

    def inside(q):  # q (absolute) as parts below rd, or None
        q = rebase_run_path(q, rd, ws)
        for r in (rd, top):
            if q is not None and q.is_relative_to(r):
                return list(q.relative_to(r).parts)
        return None

    todo, stack, links = inside(Path(p)), [], 0
    if todo is None:
        return None
    while todo:
        x = todo.pop(0)
        if x in ("", "."):
            continue
        if x == "..":
            if not stack:
                return None
            stack.pop()
            continue
        nxt = top.joinpath(*stack, x)
        if nxt.is_symlink():
            links += 1
            target = Path(os.readlink(nxt))
            if target.is_absolute():
                rest = inside(target)
                if rest is None or links > max_links:
                    return None
                todo, stack = rest + todo, []
            else:
                todo = list(target.parts) + todo
            continue
        stack.append(x)
    q = top.joinpath(*stack)
    return q if q.exists() else None


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
