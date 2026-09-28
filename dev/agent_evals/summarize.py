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

"""Per-(task, model, arm) table over every graded run of a batch. Means over reps; '-' = metric not applicable."""
import collections
import json
import statistics


def mean(xs):
    return f"{statistics.mean(xs):.2f}" if xs else "-"


def main(batch_dir):
    rows = [json.loads(p.read_text()) for p in sorted(batch_dir.glob("*/grade.json"))]
    if not rows:
        raise SystemExit(f"no graded runs under {batch_dir}")
    print(f"{batch_dir.name}: {len(rows)} runs, total ${sum(r.get('cost') or 0 for r in rows):.2f}")
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["task"], r.get("model", "-"), r["arm"])].append(r)
    cols = ["pass", "strict_pass", "format_ok", "root_cause_fixed", "recall", "kill_rate", "n_findings", "cost",
            "turns", "n_bazel", "n_build_fail"]
    print(f"{'task':24} {'model':6} {'arm':11} {'n':>2} " + " ".join(f"{c[:6]:>6}" for c in cols)
          + "  icpx bg gread decoys")
    for k in sorted(cells):
        rs = cells[k]
        vals = []
        for c in cols:
            xs = [float(r[c]) for r in rs if r.get(c) is not None and not isinstance(r[c], (list, dict))]
            vals.append(mean(xs))
        dec = collections.Counter(d for r in rs for d in (r.get("decoys_flagged") or []))
        print(f"{k[0][:24]:24} {k[1][:6]:6} {k[2][:11]:11} {len(rs):2} " + " ".join(f"{v:>6}" for v in vals)
              + f"  {sum(bool(r.get('icpx_hit')) for r in rs):4} {sum(r.get('bg_denied', 0) for r in rs):2} "
              + f"{mean([len(r.get('guidance_read', [])) for r in rs]):>5} {dict(dec) or ''}")
