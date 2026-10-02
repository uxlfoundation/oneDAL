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

"""Claude Code `--output-format stream-json` trace -> harness-independent run files.

The only module that reads the Claude format. From <run_dir>/trace.jsonl it writes

  events.jsonl  one event per tool call (agent-benchmark repository-evaluation contract, section 4.1):
                {"t", "kind": "command", "text", "rc", "denied", "output"}   Bash
                {"t", "kind": "read", "path"}                                Read
                {"t", "kind": "edit", "path"}                                Edit, Write, MultiEdit, NotebookEdit
                {"t", "kind": "tool", "name", "denied"}                      anything else
                t is seconds since the first timestamped line (null if the trace has none). Paths inside the
                agent's cwd are made relative to it. "output" is the tool result text (stdout and stderr merged,
                as Claude Code returns them); over 8 KB it keeps the first and last 4 KB around an omission line
                and "output_truncated" is true. "rc" is 0 or 1: Claude Code reports success or failure only.
  answer.txt    the final message
  usage.json    cost, turns, result subtype, and whether the CLI loaded AGENTS.md (cc-plugin-agents-md)

Usage: python3 traces.py <run_dir>
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
OUTPUT_MAX, OUTPUT_KEEP = 8192, 4096  # contract section 4.1


def _clip(body):
    """(output, truncated): over OUTPUT_MAX bytes, the first and last OUTPUT_KEEP around an omission line."""
    b = body.encode()
    if len(b) <= OUTPUT_MAX:
        return body, False
    head, tail = b[:OUTPUT_KEEP].decode(errors="ignore"), b[-OUTPUT_KEEP:].decode(errors="ignore")
    return f"{head}\n[... {len(b) - 2 * OUTPUT_KEEP} bytes omitted ...]\n{tail}", True


def _seconds(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return None


def _text(content):
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content if isinstance(content, str) else json.dumps(content)


def _rel(path, cwd):
    if not cwd or not path:
        return path
    for root, q in ((cwd, path), (os.path.realpath(cwd), os.path.realpath(path))):
        if q.startswith(root + "/"):
            return q[len(root) + 1:]
    return path


def convert(rd):
    events, pending, result = [], {}, {}
    cwd, t0, loader = None, None, None
    p = rd / "trace.jsonl"
    for line in p.read_text().splitlines() if p.exists() else []:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        now = _seconds(ev.get("timestamp"))
        t0 = now if t0 is None else t0
        t = round(now - t0, 1) if now is not None else None
        if ev.get("type") == "result":
            result = ev
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            cwd = ev.get("cwd")
            # Claude Code loads AGENTS.md through this builtin plugin; CLIs without it never load the file
            loader = any(pl.get("name") == "cc-plugin-agents-md" for pl in ev.get("plugins") or [])
        content = ev.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if ev.get("type") == "assistant" and c.get("type") == "tool_use":
                name, inp = c.get("name"), c.get("input") or {}
                if name == "Bash":
                    e = {"t": t, "kind": "command", "text": inp.get("command", ""), "rc": None, "denied": False,
                         "output": "", "output_truncated": False}
                elif name == "Read":
                    e = {"t": t, "kind": "read", "path": _rel(inp.get("file_path", ""), cwd)}
                elif name in EDIT_TOOLS:
                    e = {"t": t, "kind": "edit", "path": _rel(inp.get("file_path") or inp.get("notebook_path", ""),
                                                              cwd)}
                else:
                    e = {"t": t, "kind": "tool", "name": name, "denied": False}
                events.append(e)
                pending[c.get("id")] = e
            elif ev.get("type") == "user" and c.get("type") == "tool_result" and c.get("tool_use_id") in pending:
                e = pending.pop(c["tool_use_id"])
                body = _text(c.get("content"))
                denied = bool(c.get("is_error")) and bool(re.match(r"PreToolUse:\w+ hook error", body))
                if "denied" in e:
                    e["denied"] = denied
                if e["kind"] == "command":
                    e["output"], e["output_truncated"] = _clip(body)
                    if not denied:
                        e["rc"] = 1 if c.get("is_error") else 0
    (rd / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (rd / "answer.txt").write_text(result.get("result") or "")
    usage = {"cost": result.get("total_cost_usd"), "turns": result.get("num_turns"),
             "is_error": result.get("is_error"), "subtype": result.get("subtype"), "agents_md_loader": loader}
    (rd / "usage.json").write_text(json.dumps(usage))
    return usage


if __name__ == "__main__":
    for d in sys.argv[1:]:
        convert(Path(d))
