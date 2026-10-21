#===============================================================================
# Copyright contributors to the oneDAL project
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

"""candidates.py <build> -> out-<build>/CANDIDATES.md

Removal candidates per component/sub: tier by what still runs it.
  T1 drop: no real (A) and no daal4py-public (P) lines -> only examples/own tests or nothing
  T2 deprecate: no A, some P -> only daal4py public API keeps it (needs deprecation cycle)
  T3 trim: A exists but >=50% of lines never run by a real consumer (C+B+E) -> dead branches inside
"""
import json
import os
import sys

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
B = sys.argv[1] if len(sys.argv) > 1 else "cov2"
d = json.load(open(f"{U}/out-{B}/data.json"))
rows = []
for c in d["components"]:
    for s, v in c["subs"].items():
        L = v["lines"]; dead = v["C"] + v["B"] + v["E"]
        if L < 50: continue
        t = "T1 drop" if v["A"] + v["P"] == 0 else "T2 deprecate" if v["A"] == 0 else "T3 trim" if dead / L >= .5 else None
        if t: rows.append((t, c["api"], c["comp"], s, L, v["A"], v["P"], dead, c["sec"] * (v["lines"] / max(c["lines"], 1))))
rows.sort(key=lambda r: (r[0], -r[7]))
md = ["| tier | api | component | sub | lines | A | P | C+B+E | ~build s |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]:.0f} |" for r in rows]
open(f"{U}/out-{B}/CANDIDATES.md", "w").write("\n".join(md) + "\n")
for t in ("T1", "T2", "T3"):
    rs = [r for r in rows if r[0].startswith(t)]
    print(t, len(rs), "subs,", sum(r[7] for r in rs), "removable lines,", round(sum(r[8] for r in rs)), "build-s")
    for r in rs[:8]: print("  ", r[1], r[2], r[3], r[4], "dead", r[7], f"{r[8]:.0f}s")
