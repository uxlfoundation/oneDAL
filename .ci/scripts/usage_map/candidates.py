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
  T1i drop API only: as T1, but other algorithms include its headers or run its kernel
      (backend/primitives) -> only the public compute API goes, the kernel stays
  T2 deprecate: no A, some P -> only daal4py public API keeps it (needs deprecation cycle)
  T2s deprecate API: has P and sklearnex runs the same algorithm (here or in its oneAPI
      counterpart) -> daal4py wrapper and DAAL public API can be deprecated, the kernel stays
  T3 trim: A exists but >=50% of lines never run by a real consumer (C+B+E) -> dead branches inside
intent.json (maintainers) marks components grow/keep; those rows get verdict "keep (intent)".
"""
import csv, glob, json, os, re, sys
from collections import defaultdict
B = sys.argv[1] if len(sys.argv) > 1 else "cov2"
U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
SRC = f"{U}/oneDAL-rel/cpp"
d = json.load(open(f"out-{B}/data.json"))
intent = {k: v for k, v in json.load(open(f"{U}/intent.json")).items() if not k.startswith("_")} \
    if os.path.exists(f"{U}/intent.json") else {}

# Public-API components whose kernel lives in a shared primitive.
KERNELS = {"logloss_objective": "backend/primitives/objective_function",
           "objective_function": "backend/primitives/objective_function",
           "cosine_distance": "backend/primitives/distance", "correlation_distance": "backend/primitives/distance",
           "minkowski_distance": "backend/primitives/distance", "chebyshev_distance": "backend/primitives/distance",
           "newton_cg": "backend/primitives/optimizers"}
# DAAL component -> oneAPI component sklearnex runs for the same estimator.
COUNTERPART = {"k_nearest_neighbors": "knn", "low_order_moments": "basic_statistics", "moments": "basic_statistics",
               "cordistance": "correlation_distance", "cosdistance": "cosine_distance"}
REAL = ("sklearnex", "sklearnex-gpu", "workloads", "daal4py", "model-builders")

# Who includes a component's headers from another component's sources.
inc = defaultdict(set)
pat = {"oneapi": re.compile(r'#include\s+"oneapi/dal/algo/([a-z_]+)/'),
       "daal": re.compile(r'#include\s+"(?:src/)?algorithms/(?:kernel/)?([a-z_]+)/')}
for api, root in (("oneapi", f"{SRC}/oneapi/dal/algo"), ("daal", f"{SRC}/daal/src/algorithms")):
    for f in glob.glob(f"{root}/**/*.[ch]*", recursive=True):
        if "/test" in f or "/example" in f:
            continue
        me = os.path.relpath(f, root).split("/")[0]
        for m in pat[api].finditer(open(f, errors="ignore").read()):
            if m.group(1) != me:
                inc[(api, m.group(1))].add(me)

# Real-run lines per kernel dir, sklearnex-run lines per component.
kern = defaultdict(int); skl = defaultdict(int)
for r in csv.DictReader(open(f"out-{B}/functions.csv")):
    g = r["groups"].split("+")
    for k, p in KERNELS.items():
        if p in r["file"] and any(x in REAL for x in g):
            kern[p] += 1
    if "sklearnex" in g or "sklearnex-gpu" in g:
        skl[(r["api"], r["component"])] += 1

def internal(api, comp):
    users = sorted(u for u in inc.get((api, comp), ()) if u in {c["comp"] for c in d["components"]})
    k = KERNELS.get(comp) if api == "oneapi" else None
    return users, (k if k and kern[k] else None)

rows = []
for c in d["components"]:
    api, comp = c["api"], c["comp"]
    users, k = internal(api, comp)
    has_skl = skl[(api, comp)] > 0 or skl[("oneapi", COUNTERPART.get(comp, comp))] > 0
    for s, v in c["subs"].items():
        L = v["lines"]; dead = v["C"] + v["B"] + v["E"]
        if L < 50:
            continue
        if v["A"] + v["P"] == 0:
            t = "T1i drop API only" if users or k else "T1 drop"
        elif v["A"] == 0:
            t = "T2 deprecate"
        elif api == "daal" and not comp.startswith("shared:") and v["P"] > 0 and has_skl:
            t = "T2s deprecate API"
        elif dead / L >= .5:
            t = "T3 trim"
        else:
            continue
        note = []
        if users: note.append("included by " + ", ".join(users))
        if k: note.append(f"kernel in {k} is run")
        if t.startswith("T2s"): note.append("sklearnex runs it" if skl[(api, comp)] else f"sklearnex runs oneapi:{COUNTERPART.get(comp, comp)}")
        it = intent.get(f"{api}:{comp}")
        if it and it["intent"] in ("grow", "keep"):
            note.insert(0, f"keep (intent: {it['intent']}, {it['why']})")
        if t.startswith("T2s"):
            dead = v["P"]          # only the daal4py-API-only lines go; A lines are the kernel sklearnex runs
        rows.append((t, api, comp, s, L, v["A"], v["P"], dead, c["sec"] * (L / max(c["lines"], 1)), "; ".join(note)))
ORDER = ["T1 drop", "T1i drop API only", "T2 deprecate", "T2s deprecate API", "T3 trim"]
rows.sort(key=lambda r: (ORDER.index(r[0]), -r[7]))
md = ["Tiers: T1 drop; T1i drop public API only (kernel used elsewhere); T2 deprecate (daal4py API only);",
      "T2s deprecate daal4py/DAAL API, sklearnex covers the estimator (kernel stays, removable = P lines); T3 trim dead branches.",
      "Rows marked `keep (intent)` are planned for use (intent.json) and are not removal proposals.", "",
      "| tier | api | component | sub | lines | A | P | removable (C+B+E; P for T2s) | ~build s | note |",
      "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]:.0f} | {r[9]} |" for r in rows]
open(f"out-{B}/CANDIDATES.md", "w").write("\n".join(md) + "\n")
for t in ORDER:
    rs = [r for r in rows if r[0] == t and not r[9].startswith("keep")]
    print(t, len(rs), "subs,", sum(r[7] for r in rs), "removable lines,", round(sum(r[8] for r in rs)), "build-s")
    for r in rs[:8]: print("  ", r[1], r[2], r[3], r[4], "dead", r[7], f"{r[8]:.0f}s", r[9])
