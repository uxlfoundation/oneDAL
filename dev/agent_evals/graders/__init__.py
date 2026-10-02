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

"""Grader registry. Every module in this package exports

  GRADERS = {name: fn(rd, ws, task, trace) -> dict}  result dict; "pass" is True/False, or None for score-only
                                                    tasks. rd is the run directory (logs, output base), ws the
                                                    agent's final tree. trace = metrics.metrics(rd) (from
                                                    events.jsonl) plus "answer", the agent's final message, and
                                                    "start_sha", the prepared commit; graders never read a harness trace
  ORACLES = {name: fn(rd, repo, task) -> str}       reference answer text for the grader self-check (optional)

A task's task.json names its grader in "grader". Oracles apply the reference fix to `repo` in place and/or
return the final-answer text the grader parses.
"""
import importlib
import pkgutil

GRADERS, ORACLES = {}, {}
for _m in pkgutil.iter_modules(__path__):
    _mod = importlib.import_module(f"{__name__}.{_m.name}")
    GRADERS.update(getattr(_mod, "GRADERS", {}))
    ORACLES.update(getattr(_mod, "ORACLES", {}))
