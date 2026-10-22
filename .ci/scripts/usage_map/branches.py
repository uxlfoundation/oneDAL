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

"""branches.py <build> [--time-one] [--comps api/comp,...] -> out-<build>/BRANCHES.md + BRANCHES.json

Branch coverage of the functions that REAL consumers run. gcov is run with -b
(collect.gcov_one with GCOV="gcov -b"; without -b the JSON has no branch data).
For each object of each REAL-consumer unit, every line with branches is attributed to
the function whose start_line is the last one at or before it (same file, same object).
Owners that no real consumer runs are dropped. A branch arm is "taken" if any real unit
took it; exceptional (throw) arms are skipped. Output: never-taken arms per function.
"""
import argparse
import bisect
import concurrent.futures as cf
import glob
import gzip
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["GCOV"] = "gcov -b"          # read by collect.py at import time
import collect as C                     # noqa: E402
from report import area                 # noqa: E402

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
REAL = ["sklearnex", "sklearnex-gpu", "daal4py", "model-builders"]  # report2.py REAL
WORKERS = 24


def load(path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def gcno_list(B):
    src = f"{U}/oneDAL-{B}"
    works = [d for d in (f"{src}/__work_gnu", f"{src}/__work") if os.path.isdir(d)]
    out = []
    for w in works:
        for d, _, fs in os.walk(w):
            out += [os.path.join(d, f) for f in fs if f.endswith(".gcno") and "_static" not in d]
    return out


def process_unit(B, uid, gcnos, hit_real, acc, pool):
    """acc[(file, line, arm, owner)] = taken (OR over units and objects)"""
    gcda_root = f"{U}/cov-{B}/u/{uid}"
    for docs in pool.map(lambda g: C.gcov_one(g, gcda_root), gcnos):
        if docs is None:
            continue
        for doc in docs:
            for f in doc.get("files", []):
                fn = C.canon(f["file"])
                if fn is None:
                    continue
                fl = f.get("functions", [])
                starts = sorted((fu["start_line"], fu["name"]) for fu in fl)
                keys = [s for s, _ in starts]
                for ln in f.get("lines", []):
                    br = ln.get("branches")
                    if not br:
                        continue
                    i = bisect.bisect_right(keys, ln["line_number"]) - 1
                    if i < 0:
                        continue
                    owner = starts[i][1]
                    if owner not in hit_real:
                        continue
                    for a, b in enumerate(br):
                        if b.get("throw"):
                            continue
                        k = (fn, ln["line_number"], a, owner)
                        e = acc.get(k)
                        taken = b.get("count", 0) > 0
                        if e is None:
                            acc[k] = taken
                        elif taken:
                            acc[k] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--time-one", action="store_true", help="time one real unit and exit")
    ap.add_argument("--comps", default="", help="comma list of api/comp to restrict to")
    ap.add_argument("--units", type=int, default=0, help="limit to first N units (debug)")
    a = ap.parse_args()
    B = a.build

    uni = load(f"{U}/data/{B}/universe.json.gz")
    funcs = uni["functions"]
    units = [u for u in json.load(open(f"{U}/units-{B}.json"))
             if u["consumer"] in REAL and os.path.exists(f"{U}/data/{B}/{u['uid']}.json.gz")]
    hit_real = set()
    for u in units:
        hit_real.update(load(f"{U}/data/{B}/{u['uid']}.json.gz")["functions"])
    if a.comps:
        want = set(a.comps.split(","))
        hit_real = {m for m in hit_real if m in funcs and "/".join(area(funcs[m][0])[:2]) in want}
    if a.units:
        units = units[:a.units]
    gcnos = gcno_list(B)
    print(f"units={len(units)} gcno={len(gcnos)} real-run owners={len(hit_real)}", flush=True)

    acc = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        for n, u in enumerate(units, 1):
            t1 = time.time()
            process_unit(B, u["uid"], gcnos, hit_real, acc, pool)
            print(f"  [{n}/{len(units)}] {u['uid'][:60]} {time.time() - t1:.1f}s arms={len(acc)}", flush=True)
            if a.time_one:
                print(f"time for one unit: {time.time() - t1:.1f}s; estimated all: {(time.time() - t1) * len(units) / 60:.1f} min")
                return

    # aggregate: arms per owner, never-taken per owner
    arms = defaultdict(lambda: [0, 0])          # owner -> [arms, never-taken]
    for (fn, line, arm, owner), taken in acc.items():
        arms[owner][0] += 1
        arms[owner][1] += not taken
    comp = defaultdict(lambda: [0, 0, 0])       # api/comp -> [functions, arms, never]
    rows = []
    for m, (n, nev) in arms.items():
        f, line, dem = funcs[m]
        api, c, _ = area(f)
        k = f"{api}/{c}"
        comp[k][0] += 1
        comp[k][1] += n
        comp[k][2] += nev
        rows.append([nev, n, f, line, dem])
    rows.sort(key=lambda r: (-r[0], r[2], r[3]))

    os.makedirs(f"{U}/out-{B}", exist_ok=True)
    tot_arms = sum(v[1] for v in comp.values())
    tot_nev = sum(v[2] for v in comp.values())
    md = [f"# oneDAL branch coverage of real-consumer functions, build `{B}`\n",
          f"Units: {len(units)} real-consumer units ({', '.join(REAL)}). Functions run by them: {len(arms)}.",
          "Arm = a gcov branch arm (throw arms skipped). Taken = taken by any real unit, in any object.",
          "Owner = function by line range within the object; ISA copies count separately (per instance).",
          f"Arms: {tot_arms}, never taken: {tot_nev} ({100 * tot_nev / max(tot_arms, 1):.1f}%).\n",
          "## Top 40 functions by never-taken arms\n",
          "| never-taken | arms | file:line | function |", "|---:|---:|---|---|"]
    for nev, n, f, line, dem in rows[:40]:
        md.append(f"| {nev} | {n} | {f}:{line} | `{dem[:110].replace('|', '/')}` |")
    md += ["", "## Totals per component\n", "| component | functions | arms | never-taken | % never |", "|---|---:|---:|---:|---:|"]
    for k, (nf, n, nev) in sorted(comp.items(), key=lambda kv: -kv[1][2]):
        md.append(f"| {k} | {nf} | {n} | {nev} | {100 * nev / max(n, 1):.1f}% |")
    open(f"{U}/out-{B}/BRANCHES.md", "w").write("\n".join(md) + "\n")
    json.dump(dict(build=B, units=len(units), components={k: dict(functions=v[0], arms=v[1], never=v[2])
                                                          for k, v in comp.items()},
                   top=[dict(never=r[0], arms=r[1], file=r[2], line=r[3], name=r[4]) for r in rows[:500]]),
              open(f"{U}/out-{B}/BRANCHES.json", "w"))
    print(f"done in {time.time() - t0:.0f}s; arms={tot_arms} never={tot_nev}", flush=True)


if __name__ == "__main__":
    main()
