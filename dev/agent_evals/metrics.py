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

"""Process metrics from <run_dir>/events.jsonl, the harness-independent trace agent-benchmark writes."""
import json
import re

BAZEL = re.compile(r"\bbazel(?:isk)?\b")
MAKE = re.compile(r"\bmake\b")
BUILD_FAILED = re.compile(r"FAILED TO BUILD|Build did NOT complete|\bFAILED\b|Error \d+")


def read_events(rd):
    p = rd / "events.jsonl"
    out = []
    for line in p.read_text().splitlines() if p.exists() else []:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def is_guidance(path):
    return path.endswith(("AGENTS.md", "CLAUDE.md")) or "/.github/instructions/" in "/" + path


def failed(e):
    return e.get("denied") or e.get("rc") not in (0, None) or bool(BUILD_FAILED.search(e.get("output") or ""))


def metrics(rd):
    """bazel_cmds is every Bash command naming bazel, refused ones included (as in the trace parser it replaces)."""
    ev = read_events(rd)
    cmds = [e for e in ev if e.get("kind") == "command"]
    bazel_ev = [e for e in cmds if BAZEL.search(e.get("text", ""))]
    make_ev = [e for e in cmds if not BAZEL.search(e.get("text", "")) and MAKE.search(e.get("text", ""))]
    build_ev = bazel_ev + make_ev
    return {"tool_calls": len(ev), "bazel_cmds": [e["text"] for e in bazel_ev], "n_bazel": len(bazel_ev),
            "n_make": len(make_ev), "n_bazel_fail": sum(map(failed, bazel_ev)),
            "n_build_fail": sum(map(failed, build_ev)),
            "icpx_hit": any("icpx is not found" in (e.get("output") or "") for e in build_ev),
            "guidance_read": sorted({e["path"] for e in ev if e.get("kind") == "read" and is_guidance(e["path"])}),
            "bg_denied": sum(bool(e.get("denied")) for e in cmds)}
