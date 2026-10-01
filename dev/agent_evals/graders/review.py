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

"""Review tasks: the agent reviews HEAD and reports findings as JSON; matched against a manifest.

manifest.json:
  seeds   [{id, class, file, lines, kw}]  known problems. A finding hits a seed if its file matches ("*" = any) and
                                          its rule/message matches kw, or (no kw hit) its line is within 1 of lines;
                                          the line fallback credits only the nearest such seed.
  decoys  [{id, file, lines, kw}]         things that look wrong but are fine. Flagging one is a false positive.
A manifest with no seeds is a clean-diff control: every finding is a false positive.
"""
import json
import re

from common import last_json, task_dir

SEVERE = ("blocker", "major")


def norm_path(p):
    """Repository-relative path of a finding: absolute paths into the run's repo/ are cut at repo/."""
    p = str(p or "").strip().replace("\\", "/")
    if "/repo/" in p:
        p = p.rsplit("/repo/", 1)[1]
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def finding_fields(f):
    path = norm_path(f.get("file"))
    text = " ".join(str(f.get(k, "")) for k in ("rule", "message", "evidence"))
    try:
        line = int(f.get("line") or -99)
    except (TypeError, ValueError):
        line = -99
    return path, text, line


def same_file(entry, path):
    return entry["file"] == "*" or bool(path) and path == norm_path(entry["file"])


def g_review(rd, t, tr):
    man = json.loads((task_dir(t["id"]) / t.get("manifest", "manifest.json")).read_text())
    seeds, decoys = man.get("seeds", []), man.get("decoys", [])
    findings = last_json(tr["answer"])
    if not isinstance(findings, list):
        return {"pass": False, "parse_error": True}
    findings = [f for f in findings if isinstance(f, dict)]
    hit, fp, other = set(), set(), 0
    for f in findings:
        path, text, line = finding_fields(f)
        matched = [s["id"] for s in seeds if same_file(s, path) and re.search(s["kw"], text, re.I)]
        if not matched:  # keyword miss: fall back to line proximity, crediting only the nearest seed
            near = [(min(abs(line - x) for x in s["lines"]), s["id"]) for s in seeds
                    if same_file(s, path) and s["file"] != "*" and s["lines"]]
            near = [n for n in near if n[0] <= 1]
            matched = [min(near)[1]] if near else []
        if matched:
            hit.update(matched)
            continue
        dec = [d["id"] for d in decoys if same_file(d, path) and (re.search(d["kw"], text, re.I) or line in d["lines"])]
        if dec:
            fp.update(dec)
        else:
            other += 1
    by_class = {}
    for s in seeds:
        c = by_class.setdefault(s["class"], [0, 0])
        c[1] += 1
        c[0] += s["id"] in hit
    n_severe = sum(str(f.get("severity", "")).lower() in SEVERE for f in findings)
    out = {"pass": None, "n_findings": len(findings), "n_severe": n_severe, "seeds_hit": sorted(hit),
           "decoys_flagged": sorted(fp), "n_decoys": len(decoys), "unmatched": other,
           "recall": round(len(hit) / len(seeds), 3) if seeds else None,
           "recall_by_class": {k: f"{a}/{b}" for k, (a, b) in by_class.items()}}
    if not seeds:  # clean control: pass = nothing blocking reported
        out["pass"] = n_severe == 0
    return out


def o_review(rd, repo, t):
    """Reference answer from oracle.json (list of findings) next to the manifest."""
    return "```json\n" + (task_dir(t["id"]) / t.get("oracle_answer", "oracle.json")).read_text() + "\n```"


GRADERS = {"review": g_review}
ORACLES = {"review": o_review}
