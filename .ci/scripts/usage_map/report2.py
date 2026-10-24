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

"""report2.py <build> -> out-<build>/{data.json,functions.csv,REPORT.md}

Per-unit coverage (collect.py), static call graph (callgraph.py), build cost
(cost.py on the release build) and test time (run_units.py) joined per
component. Classes per function group:
  A  executed by a real consumer test (sklearnex/onedal, daal4py, model builders)
  P  executed only by daal4py public-API tests
  C  executed only by oneDAL examples, oneDAL's own Bazel tests or the onedal Python package tests
     (filtered consumers: product-binding tests, not product usage)
  B  never executed, but a direct call path from A code exists (static)
  E  never executed, no static path from A code
A function group folds the per-ISA instantiations (CpuType sse2/avx2/avx512)
of one source function: they are one piece of code compiled N times.
"""
import csv
import glob
import gzip
import json
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usage_consts import CPU, FILTERED, PUBLIC, REAL, U, area, load_units, static_tag  # noqa: E402

B = sys.argv[1]
OUT = f"{U}/out-{B}"
GROUPS = REAL + PUBLIC + FILTERED
LIBDIR = f"{U}/oneDAL-{B}/__release_lnx_gnu/daal/latest/lib/intel64"


def load(path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def exported():
    names = set()
    for so in os.listdir(LIBDIR):
        if so.endswith(".so.4.0"):
            r = subprocess.run(["nm", "-D", "--defined-only", os.path.join(LIBDIR, so)],
                               capture_output=True, text=True)
            names.update(ln.split()[-1] for ln in r.stdout.splitlines() if ln.split()[1:2] in (["T"], ["W"]))
    return names


_load_time = {}


def load_time(f):
    """Lines of f that run at library load (serialization registration).

    llvm-cov reports the registrar functions these macros expand to as executed by every
    GPU run, so they would mark any algorithm as used. gcc attributes them elsewhere.
    """
    if f not in _load_time:
        try:
            src = open(f"{U}/oneDAL-gpu/{f}", errors="replace").read().splitlines()
        except OSError:
            src = []
        reg = "__DAAL_REGISTER_SERIALIZATION_CLASS"
        _load_time[f] = {i + 1 for i, t in enumerate(src) if t.lstrip().startswith(reg)}
    return _load_time[f]


def main():
    uni = load(f"{U}/data/{B}/universe.json.gz")
    funcs = uni["functions"]                   # mangled -> [file, line, demangled]
    ulines = {f: set(v) for f, v in uni["lines"].items()}
    units = {u["uid"]: u for u in load_units(B)}
    hit = defaultdict(set)                     # group -> mangled executed
    uline = {}                                 # uid -> {file: set(lines)}
    gline = {g: defaultdict(set) for g in GROUPS}
    for uid, u in units.items():
        p = f"{U}/data/{B}/{uid}.json.gz"
        if not os.path.exists(p):
            continue
        d = load(p)
        hit[u["consumer"]].update(d["functions"])
        uline[uid] = {f: set(v) for f, v in d["lines"].items()}
        for f, v in uline[uid].items():
            gline[u["consumer"]][f] |= set(v)

    # GPU build (collect-gpu.py): host-side hits of the sklearnex/onedal GPU and SPMD runs.
    # llvm-cov names functions by demangled name only, so they join on (file, normalized name).
    gpu_fn = {}                                # (file, normalized) -> GPU-only record source
    if os.path.exists(f"{U}/data/gpu/universe.json.gz"):
        guni = load(f"{U}/data/gpu/universe.json.gz")
        for f, v in guni["lines"].items():
            ulines.setdefault(f, set()).update(v)
        gsec = {x["uid"]: x for x in json.load(open(f"{U}/units-gpu.json"))} if os.path.exists(
            f"{U}/units-gpu.json") else {}
        for p in sorted(glob.glob(f"{U}/data/gpu/gpu_*.json.gz")):
            uid = os.path.basename(p)[:-8]
            d = load(p)
            units[uid] = dict(uid=uid, id=f"gpu:{uid[4:]}", consumer="sklearnex-gpu",
                              seconds=gsec.get(uid, {}).get("seconds", 0), rc=gsec.get(uid, {}).get("rc", 0))
            uline[uid] = {f: set(v) - load_time(f) for f, v in d["lines"].items()}
            for f, v in uline[uid].items():
                gline["sklearnex-gpu"][f] |= set(v)
            for name in d["functions"]:
                if name in guni["functions"]:
                    f, line, dem = guni["functions"][name]
                    if line in load_time(f):
                        continue
                    gpu_fn.setdefault((f, CPU.sub("CPU", dem)), [f, line, True])
        for name, (f, line, dem) in guni["functions"].items():
            gpu_fn.setdefault((f, CPU.sub("CPU", dem)), [f, line, False])

    a_set = set().union(*(hit[g] for g in REAL))
    # static reach from executed A code over direct calls
    cg = load(f"{U}/data/{B}/callgraph.json.gz")
    reach, stack = set(a_set), list(a_set)
    while stack:
        for c in cg.get(stack.pop(), ()):
            if c not in reach:
                reach.add(c)
                stack.append(c)

    exp = exported()
    grp = {}                                   # (file, normalized) -> record
    for m, (f, line, dem) in funcs.items():
        key = (f, CPU.sub("CPU", dem))
        r = grp.setdefault(key, dict(file=f, line=line, name=key[1], inst=0, used=set(), reach=False, abi=False))
        r["inst"] += 1
        r["line"] = min(r["line"], line)
        r["used"].update(g for g in GROUPS if m in hit[g])
        r["reach"] |= m in reach
        r["abi"] |= m in exp
    at = {}                                    # (file, line) -> CPU group, for names llvm-cov spells differently
    for m, (f, line, dem) in funcs.items():
        at.setdefault((f, line), grp[(f, CPU.sub("CPU", dem))])
    for key, (f, line, ran) in gpu_fn.items():
        # GPU-only functions have no call graph: never-run ones count as E
        r = grp.get(key) or at.get((f, line)) or grp.setdefault(key, dict(file=f, line=line, name=key[1], inst=1, used=set(), reach=False, abi=False))
        if ran:
            r["used"].add("sklearnex-gpu")
    for r in grp.values():
        u = r["used"]
        r["cls"] = ("A" if u & set(REAL) else "P" if u & set(PUBLIC) else "C" if u & set(FILTERED)
                    else "B" if r["reach"] else "E")

    # line classes per file: A/P/C hit by group; B/E for never-hit lines by owning function
    fn_ranges = defaultdict(list)
    for r in grp.values():
        fn_ranges[r["file"]].append((r["line"], r["cls"]))
    files = {}
    for f, lines in ulines.items():
        a = set().union(*(gline[g].get(f, set()) for g in REAL)) & lines
        p = set().union(*(gline[g].get(f, set()) for g in PUBLIC)) & lines - a
        c = set().union(*(gline[g].get(f, set()) for g in FILTERED)) & lines - a - p
        rest = sorted(lines - a - p - c)
        starts = sorted(fn_ranges.get(f, []))
        nb = 0
        j = -1
        for ln in rest:   # owner = last function starting at or before the line
            while j + 1 < len(starts) and starts[j + 1][0] <= ln:
                j += 1
            nb += j >= 0 and starts[j][1] in "ABPC"
        api, comp, sub = area(f)
        files[f] = dict(api=api, comp=comp, sub=sub, lines=len(lines), A=len(a), P=len(p), C=len(c),
                        B=nb, E=len(rest) - nb,
                        by={g: len(gline[g].get(f, set()) & lines) for g in GROUPS})

    # build cost (release build) per source file
    cost = defaultdict(lambda: dict(sec=0.0, sec_static=0.0, bytes=0, bytes_static=0, tus=0))
    for t in json.load(open(f"{U}/data/cost-rel.json")):
        f = os.path.normpath(t["src"])
        st = "static" in t["lib"]
        c = cost[f]
        c["sec_static" if st else "sec"] += t["sec"]
        c["bytes_static" if st else "bytes"] += t["bytes"]
        c["tus"] += not st

    comps = {}
    for f, x in files.items():
        k = f"{x['api']}/{x['comp']}"
        c = comps.setdefault(k, dict(api=x["api"], comp=x["comp"], static=static_tag(x["api"], x["comp"]),
                                     lines=0, A=0, P=0, C=0, B=0, E=0, by={g: 0 for g in GROUPS},
                                     fn={c_: 0 for c_ in "APCBE"}, fn_abi=0, files=0, subs={},
                                     sec=0.0, sec_static=0.0, bytes=0, bytes_static=0, tus=0, units={}))
        c["files"] += 1
        for kk in ("lines", "A", "P", "C", "B", "E"):
            c[kk] += x[kk]
            s = c["subs"].setdefault(x["sub"], dict(lines=0, A=0, P=0, C=0, B=0, E=0))
            s[kk] += x[kk]
        for g in GROUPS:
            c["by"][g] += x["by"][g]
    for f, cc in cost.items():
        api, comp, _ = area(f) if f.startswith("cpp/") else ("other", "shared:other", "-")
        c = comps.get(f"{api}/{comp}")
        if c:
            for kk in ("sec", "sec_static", "bytes", "bytes_static", "tus"):
                c[kk] += cc[kk]
    for r in grp.values():
        api, comp, _ = area(r["file"])
        c = comps[f"{api}/{comp}"]
        c["fn"][r["cls"]] += 1
        c["fn_abi"] += r["abi"] and r["cls"] != "A"
    # tests touching a component: the cost of re-testing it after a change
    for uid, fl in uline.items():
        touched = defaultdict(int)
        for f, ls in fl.items():
            if f in files:
                x = files[f]
                touched[f"{x['api']}/{x['comp']}"] += len(ls & ulines[f])
        tot = sum(touched.values()) or 1
        for k, n in touched.items():
            if n:
                comps[k]["units"][uid] = n
                c = comps[k]
                # share of this unit's time spent on k, by hit-line share
                c["test_sec_attr"] = c.get("test_sec_attr", 0) + units[uid]["seconds"] * n / tot
                c["focused"] = c.get("focused", 0) + (n / tot >= 0.10)
    for c in comps.values():
        c["test_sec_attr"] = round(c.get("test_sec_attr", 0), 1)
        c.setdefault("focused", 0)
        c["test_sec"] = round(sum(units[u]["seconds"] for u in c["units"]), 1)
        c["test_sec_real"] = round(sum(units[u]["seconds"] for u in c["units"]
                                       if units[u]["consumer"] in REAL + PUBLIC), 1)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/functions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "abi_exported", "api", "component", "sub", "file", "line", "instances", "function", "groups"])
        for r in sorted(grp.values(), key=lambda r: (r["file"], r["line"])):
            api, comp, sub = area(r["file"])
            w.writerow([r["cls"], int(r["abi"]), api, comp, sub, r["file"], r["line"], r["inst"], r["name"],
                        "+".join(sorted(r["used"]))])
    top_fn = defaultdict(list)
    for r in grp.values():
        api, comp, _ = area(r["file"])
        top_fn[f"{api}/{comp}"].append([r["cls"], r["file"], r["line"], r["name"][:160], "+".join(sorted(r["used"]))])
    data = dict(build=B, groups=GROUPS, real=REAL, public=PUBLIC, filtered=FILTERED,
                units=[dict(uid=u["uid"], id=u["id"], consumer=u["consumer"], seconds=u["seconds"], rc=u["rc"])
                       for u in units.values()],
                components=[dict(c, key=k, subs=c["subs"], units=c["units"],
                                 functions=sorted(top_fn[k], key=lambda x: ("EBCPA".index(x[0]), x[1], x[2]))[:400])
                            for k, c in comps.items()],
                files=[dict(x, file=f) for f, x in files.items()])
    json.dump(data, open(f"{OUT}/data.json", "w"))

    T = defaultdict(float)
    for c in comps.values():
        for kk in ("lines", "A", "P", "C", "B", "E", "sec", "sec_static", "bytes", "bytes_static"):
            T[kk] += c[kk]
    F = defaultdict(int)
    for r in grp.values():
        F[r["cls"]] += 1
    L = T["lines"]
    md = [f"# oneDAL 2026.1.0 usage map, build `{B}` (O0, sse2+avx2+avx512, headers mapped)\n",
          f"Function groups (ISA instances folded): {len(grp)} from {len(funcs)} symbols. "
          + "  ".join(f"{k}={F[k]}" for k in "APCBE") + "\n",
          "Lines: %d  A=%.1f%%  P=%.1f%%  C=%.1f%%  B=%.1f%%  E=%.1f%%\n" % (
              L, *(100 * T[k] / L for k in "APCBE")),
          "Build (release flags): %.0f CPU-s dynamic + %.0f CPU-s static duplicate; objects %.1f MB + %.1f MB\n" % (
              T["sec"], T["sec_static"], T["bytes"] / 1e6, T["bytes_static"] / 1e6),
          "\n| component | api | static | lines | A | P | C | B | E | build CPU-s | obj KB | test-s attributed | focused tests | test-s touching (consumers) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, c in sorted(comps.items(), key=lambda kv: -(kv[1]["E"] + kv[1]["B"] + kv[1]["C"])):
        md.append(f"| {c['comp']} | {c['api']} | {c['static']} | {c['lines']} | {c['A']} | {c['P']} | {c['C']} | "
                  f"{c['B']} | {c['E']} | {c['sec']:.0f} | {c['bytes'] / 1e3:.0f} | {c['test_sec_attr']:.0f} | {c['focused']} | {c['test_sec_real']:.0f} |")
    open(f"{OUT}/REPORT.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[:4]))


if __name__ == "__main__":
    main()
