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

"""Review-replay tasks: a real merged PR replayed at the revision humans reviewed; gold = their review threads.

gold.json (next to task.json, or t["gold"]):
  [{id, file, line, kind, comment, addressed}]          one human thread with a line anchor on the reviewed
                                                        revision. line is in that revision's new file; kind is
                                                        correctness|api|tests|convention|docs|nit; addressed says
                                                        whether the code was changed in response before merge.
A thread is hit if a finding is on the same file and |finding.line - gold.line| <= t.get("window", 5). Matching is
one-to-one: pairs are taken nearest first, so one finding credits at most one thread and vice versa. Location is the
only criterion on purpose (no judge, deterministic); recall is a score, not a verdict.
Score-only (pass None): humans miss things and raise non-defects, so neither recall nor precision is a verdict.
precision_proxy = findings within the window of any gold thread / n_findings (a lower bound on precision).
"recall" (= recall_all) and "decoys_flagged" ([]) are emitted so run.py's review self-check applies unchanged.
"""
import json

from common import last_json, task_dir
from graders.review import SEVERE, finding_fields, same_file


def _near(g, path, line, window):
    return same_file(g, path) and abs(line - g["line"]) <= window


def _ratio(a, b):
    return round(a / b, 3) if b else None


def g_replay(rd, ws, t, tr):
    gold = json.loads((task_dir(t["id"]) / t.get("gold", "gold.json")).read_text())
    window = t.get("window", 5)
    findings = last_json(tr["answer"])
    if not isinstance(findings, list):
        return {"pass": False, "parse_error": True}
    findings = [f for f in findings if isinstance(f, dict)]
    pairs, near = [], 0
    for i, f in enumerate(findings):
        path, _, line = finding_fields(f)
        close = [(abs(line - g["line"]), i, g["id"]) for g in gold if _near(g, path, line, window)]
        pairs += close
        near += bool(close)
    hit, used = set(), set()
    for _, i, gid in sorted(pairs):
        if i not in used and gid not in hit:
            used.add(i)
            hit.add(gid)
    by_kind = {}
    for g in gold:
        c = by_kind.setdefault(g["kind"], [0, 0])
        c[1] += 1
        c[0] += g["id"] in hit
    addressed = [g for g in gold if g.get("addressed")]
    recall_all = _ratio(len(hit), len(gold))
    return {"pass": None, "n_findings": len(findings),
            "n_severe": sum(str(f.get("severity", "")).lower() in SEVERE for f in findings),
            "threads_hit": sorted(hit), "n_threads": len(gold),
            "recall_all": recall_all, "recall": recall_all,
            "recall_addressed": _ratio(sum(g["id"] in hit for g in addressed), len(addressed)),
            "recall_by_kind": {k: f"{a}/{b}" for k, (a, b) in by_kind.items()},
            "precision_proxy": _ratio(near, len(findings)), "decoys_flagged": []}


def o_replay(rd, repo, t):
    """Reference answer: one finding per gold thread, at its anchor."""
    gold = json.loads((task_dir(t["id"]) / t.get("gold", "gold.json")).read_text())
    ans = [{"file": g["file"], "line": g["line"], "severity": "minor", "rule": g["kind"],
            "message": g["comment"].splitlines()[0][:200]} for g in gold]
    return "```json\n" + json.dumps(ans, indent=1) + "\n```"


GRADERS = {"review_replay": g_replay}
ORACLES = {"review_replay": o_replay}
