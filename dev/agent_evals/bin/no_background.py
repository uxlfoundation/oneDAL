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

"""PreToolUse hook: refuse backgrounded shell commands in headless eval runs.

`claude -p` exits when the model ends its turn, so a build left running in the background is killed and the
run is graded as a failure that has nothing to do with the task. Denials are logged to $EVAL_RUN_DIR/hook.log.
"""
import json
import os
import re
import sys

ev = json.load(sys.stdin)
inp = ev.get("tool_input") or {}
cmd = inp.get("command", "")
bad = None
if inp.get("run_in_background"):
    bad = "run_in_background"
elif re.search(r"(?<![&|>])&(?![&>])|\bnohup\b|\bsetsid\b|\bdisown\b", re.sub(r"'[^']*'|\"[^\"]*\"", "", cmd)):
    bad = "shell background"
if bad:
    rd = os.environ.get("EVAL_RUN_DIR")
    if rd:
        with open(os.path.join(rd, "hook.log"), "a") as f:
            f.write(json.dumps({"denied": bad, "command": cmd[:500]}) + "\n")
    print(f"Background commands are disabled in this environment ({bad}). Run the command in the foreground; "
          "long builds are fine, the tool timeout can be raised with the timeout parameter.", file=sys.stderr)
    sys.exit(2)
