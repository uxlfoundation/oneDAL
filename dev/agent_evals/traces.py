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

"""Process metrics from a `claude -p --output-format stream-json` trace."""
import json
import re


def parse_trace(rd):
    res, tools, bazel_cmds, make_cmds, read_files = None, 0, [], [], []
    build_ids, build_fail, icpx_hit = set(), 0, False
    p = rd / "trace.jsonl"
    for line in p.read_text().splitlines() if p.exists() else []:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            res = ev
        content = ev.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if ev.get("type") == "assistant" and c.get("type") == "tool_use":
                tools += 1
                inp = c.get("input", {})
                cmd = inp.get("command", "") if c["name"] == "Bash" else ""
                if re.search(r"\bbazel\b", cmd):
                    bazel_cmds.append(cmd)
                    build_ids.add(c.get("id"))
                elif re.search(r"\bmake\b", cmd):
                    make_cmds.append(cmd)
                    build_ids.add(c.get("id"))
                if c["name"] == "Read":
                    read_files.append(inp.get("file_path", ""))
            if ev.get("type") == "user" and c.get("type") == "tool_result" and c.get("tool_use_id") in build_ids:
                body = json.dumps(c.get("content"))
                if c.get("is_error") or re.search(r"FAILED TO BUILD|Build did NOT complete|\bFAILED\b|Error \d+", body):
                    build_fail += 1
                icpx_hit |= "icpx is not found" in body
    guidance_read = sorted({f.split("/repo/")[-1] for f in read_files
                            if f.endswith(("AGENTS.md", "CLAUDE.md")) or "/.github/instructions/" in f})
    hook = rd / "hook.log"
    return {"result_text": (res or {}).get("result", ""), "cost": (res or {}).get("total_cost_usd"),
            "turns": (res or {}).get("num_turns"), "is_error": (res or {}).get("is_error"),
            "subtype": (res or {}).get("subtype"), "tool_calls": tools,
            "bazel_cmds": bazel_cmds, "n_bazel": len(bazel_cmds), "n_make": len(make_cmds),
            "n_build_fail": build_fail, "icpx_hit": icpx_hit, "guidance_read": guidance_read,
            "bg_denied": len(hook.read_text().splitlines()) if hook.exists() else 0}
