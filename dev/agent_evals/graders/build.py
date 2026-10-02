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

"""Build tasks: the agent reports a command or an artifact; the grader re-runs or inspects it."""
import re
import shlex
from pathlib import Path

from common import bazel, last_json, resolve_in_run


def g_bazel_cmd(rd, ws, t, tr):
    """Re-run the single `bazel test` command the agent reported; the expected target must PASS."""
    j = last_json(tr["answer"]) or {}
    cmd = (j.get("command") or "").strip()
    out = {"reported_command": cmd}
    if not re.match(r"^bazel\s+test\b", cmd) or any(c in cmd for c in ";&|`$<>\r\n"):
        return {**out, "pass": False, "why": "no/invalid bazel test command"}
    try:
        shlex.split(cmd)
    except ValueError:
        return {**out, "pass": False, "why": "unparseable bazel test command"}
    rc, log = bazel(rd, ws, cmd[len("bazel"):] + " --nocache_test_results", rd / "grade.log")
    ok = rc == 0 and re.search(re.escape(t["expect_target"]) + r"\s+PASSED", log) is not None
    return {**out, "pass": ok, "rerun_rc": rc}


def g_artifact_path(rd, ws, t, tr):
    """Reported path must be a real ELF file named like t["artifact_re"], built inside this run's directory.

    t["path_must_contain"] is a substring the unresolved path must contain (the release layout directory).
    The path is resolved inside rd (resolve_in_run), so a run copied or archived elsewhere grades the same.
    """
    j = last_json(tr["answer"]) or {}
    p = Path(j.get("path") or "/nonexistent")
    real = resolve_in_run(p, rd, ws)
    ok = bool(real and real.is_file() and re.match(t.get("artifact_re", r"libonedal_core\.so"), real.name)
              and t.get("path_must_contain", "release") in str(p) and open(real, "rb").read(4) == b"\x7fELF")
    return {"reported_path": str(p), "reported_command": j.get("command"), "pass": ok}


def o_bazel_cmd(rd, repo, t):
    return '```json\n{"command": "bazel test ' + t["expect_target"] + '"}\n```'


def o_release_path(rd, repo, t):
    bazel(rd, repo, "build //:release --release_dpc=false", rd / "oracle_build.log")
    lib = next(iter(sorted((repo / "bazel-bin/release").rglob("libonedal_core.so*"))), None)
    return '```json\n{"path": "' + str(lib) + '"}\n```'


GRADERS = {"bazel_cmd": g_bazel_cmd, "release_path": g_artifact_path, "artifact_path": g_artifact_path}
ORACLES = {"bazel_cmd": o_bazel_cmd, "release_path": o_release_path}
