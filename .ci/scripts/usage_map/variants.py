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

"""variants.py <build> -> out-<build>/VARIANTS.md + VARIANTS.json

Template/variant usage at the instance (mangled symbol) level. Each instance of the
universe is parsed from its demangled name into three axes:
  fptype  float / double / mixed (both tokens in template args) / none (no fp template arg)
  method  <Algorithm>::Method=<value> from the `(ns::Method)N` enum casts ("+"-joined if several)
  ISA     CpuType value (sse2 / avx2 / avx512, from `(daal::internal::CpuType)N`
          or `cpu_dispatch_<isa>` in oneapi names); "none" if the name has no ISA axis.
Run = executed by at least one REAL consumer (REAL list copied from report2.py).

Two counts are reported per component:
  instances  mangled symbols; the ISA axis multiplies them (one copy per ISA)
  groups     report2.py groups: instances folded over the ISA axis (CPU token normalized)

The build was run on ONE x86 CPU, so only one ISA branch executes at runtime. The ISA
breakdown therefore means "ISA copies built"; the run column for ISA is only what this host ran.
"""
import gzip
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import area  # noqa: E402

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
B = sys.argv[1] if len(sys.argv) > 1 else "cov2"
OUT = f"{U}/out-{B}"
def load_units():
    """units-<build>.json plus scikit-learn_bench cases (units-bench.json), as in report2.py."""
    us = json.load(open(f"{U}/units-{B}.json"))
    if os.path.exists(f"{U}/units-bench.json"):
        us += json.load(open(f"{U}/units-bench.json"))
    return us


REAL = ["sklearnex", "sklearnex-gpu", "workloads", "daal4py", "model-builders"]  # report2.py REAL
CPU = re.compile(r"\((?:daal::)?(?:internal::)?CpuType\)\d+|\b(?:sse2|sse42|avx2|avx512)\b")
ISA_NAME = {"0": "sse2", "4": "avx2", "6": "avx512"}
ENUM_METHOD = re.compile(r"\(((?:\w+::)*?(\w+)::Method)\)(\d+)")
ISA_CAST = re.compile(r"\(\s*(?:daal::)?(?:internal::)?CpuType\)(\d+)")
ISA_DISPATCH = re.compile(r"cpu_dispatch_(\w+)")


def load(path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def strip_parens(s):
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"\([^()]*\)", "", s)
    return s


def fptype(dem):
    """float/double seen inside template argument lists (parameter lists removed)."""
    s = strip_parens(dem)
    depth, inside = 0, []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif depth:
            inside.append(ch)
        if depth and ch in "<>":
            inside.append(" ")
    toks = set(re.findall(r"\b(float|double)\b", "".join(inside)))
    if len(toks) == 2:
        return "mixed"
    return toks.pop() if toks else "none"


def method(dem):
    ms = [f"{alg}::Method={v}" for _, alg, v in ENUM_METHOD.findall(dem)]
    return "+".join(sorted(set(ms))) if ms else "none"


def isa(dem):
    m = ISA_CAST.findall(dem)
    if m:
        return ISA_NAME.get(m[0], f"cpu{m[0]}")
    m = ISA_DISPATCH.findall(dem)
    if m:
        return m[0]
    return "none"


def group_key(f, dem):
    return (f, CPU.sub("CPU", dem))


def main():
    uni = load(f"{U}/data/{B}/universe.json.gz")
    funcs = uni["functions"]
    units = load_units()
    hit_real = set()
    n_real_units = 0
    for u in units:
        if u["consumer"] not in REAL:
            continue
        p = f"{U}/data/{B}/{u['uid']}.json.gz"
        if os.path.exists(p):
            n_real_units += 1
            hit_real.update(load(p)["functions"])

    comps = {}

    def comp(f):
        api, c, _ = area(f)
        k = f"{api}/{c}"
        if k not in comps:
            comps[k] = dict(
                key=k, api=api, comp=c, instances=0, run=0,
                fp=defaultdict(lambda: [0, 0]), isa=defaultdict(lambda: [0, 0]),
                meth=defaultdict(lambda: [0, 0]), groups=set(), groups_run=set(),
                fp_total=None)
        return comps[k]

    gseen = {}
    for m, (f, line, dem) in funcs.items():
        c = comp(f)
        ran = m in hit_real
        fp, me, ia = fptype(dem), method(dem), isa(dem)
        c["instances"] += 1
        c["run"] += ran
        for key, val in (("fp", fp), ("meth", me), ("isa", ia)):
            cell = c[key][val]
            cell[0] += 1
            cell[1] += ran
        gk = group_key(f, dem)
        c["groups"].add(gk)
        if ran:
            c["groups_run"].add(gk)

    # JSON-serializable copies
    out = []
    for k, c in comps.items():
        out.append(dict(
            key=k, api=c["api"], comp=c["comp"], instances=c["instances"], run=c["run"],
            never=c["instances"] - c["run"], groups=len(c["groups"]), groups_run=len(c["groups_run"]),
            fptype={v: dict(total=t, run=r) for v, (t, r) in sorted(c["fp"].items())},
            isa_copies={v: dict(total=t, run=r) for v, (t, r) in sorted(c["isa"].items())},
            method={v: dict(total=t, run=r) for v, (t, r) in sorted(c["meth"].items())},
        ))
    out.sort(key=lambda x: -x["never"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(build=B, real=REAL, real_units=n_real_units, instances=len(funcs),
                   run_instances=len(hit_real & set(funcs)), components=out),
              open(f"{OUT}/VARIANTS.json", "w"))

    T_inst = sum(x["instances"] for x in out)
    T_run = sum(x["run"] for x in out)
    fp_tot = defaultdict(lambda: [0, 0])
    isa_tot = defaultdict(lambda: [0, 0])
    for x in out:
        for v, d in x["fptype"].items():
            fp_tot[v][0] += d["total"]; fp_tot[v][1] += d["run"]
        for v, d in x["isa_copies"].items():
            isa_tot[v][0] += d["total"]; isa_tot[v][1] += d["run"]
    G_tot = sum(x["groups"] for x in out)
    G_run = sum(x["groups_run"] for x in out)

    md = [f"# oneDAL template/variant usage, build `{B}`\n",
          f"Consumers counted as real: {', '.join(REAL)} ({n_real_units} units with data).",
          "Run = executed by a real consumer. ISA column = ISA copies built; the build ran on one CPU,",
          "so only one ISA copy executes at runtime (the run count per ISA is what this host happened to execute).\n",
          f"Instances (mangled symbols): {T_inst}, run by real consumers: {T_run} ({100*T_run/max(T_inst,1):.1f}%).",
          f"Groups (ISA folded): {G_tot}, run by real: {G_run} ({100*G_run/max(G_tot,1):.1f}%).\n",
          "## Totals by axis (instances, run)\n",
          "| fptype | total | run | % run |", "|---|---:|---:|---:|"]
    for v, (t, r) in sorted(fp_tot.items()):
        md.append(f"| {v} | {t} | {r} | {100*r/max(t,1):.1f}% |")
    md += ["", "| ISA copy | total | run | % run |", "|---|---:|---:|---:|"]
    for v, (t, r) in sorted(isa_tot.items()):
        md.append(f"| {v} | {t} | {r} | {100*r/max(t,1):.1f}% |")
    md += ["", "## Per component\n",
           "| component | instances | run | never | groups | groups run | float tot/run | double tot/run | mixed/none tot/run | ISA copies sse2/avx2/avx512 (run) |",
           "|---|---:|---:|---:|---:|---:|---|---|---|---|"]
    def g(x, key, v):
        d = x[key].get(v, dict(total=0, run=0))
        return f"{d['total']}/{d['run']}"
    for x in out:
        other_t = sum(d["total"] for v, d in x["fptype"].items() if v in ("mixed", "none"))
        other_r = sum(d["run"] for v, d in x["fptype"].items() if v in ("mixed", "none"))
        isa_s = "/".join(f"{x['isa_copies'].get(v, dict(total=0))['total']}({x['isa_copies'].get(v, dict(run=0))['run']})"
                         for v in ("sse2", "avx2", "avx512"))
        md.append(f"| {x['key']} | {x['instances']} | {x['run']} | {x['never']} | {x['groups']} | {x['groups_run']} | "
                  f"{g(x,'fptype','float')} | {g(x,'fptype','double')} | {other_t}/{other_r} | {isa_s} |")

    md += ["", "## Top 30 components by never-run instances\n",
           "| component | never-run | instances | run | % never | never-run groups |", "|---|---:|---:|---:|---:|---:|"]
    for x in out[:30]:
        md.append(f"| {x['key']} | {x['never']} | {x['instances']} | {x['run']} | "
                  f"{100*x['never']/max(x['instances'],1):.0f}% | {x['groups']-x['groups_run']} |")

    md += ["", "## Methods no real consumer runs\n",
           "A method is `<algorithm>::Method=<value>` from the enum casts in the name. Listed when",
           "instances exist and none of them was run by a real consumer.\n",
           "| component | method | instances |", "|---|---|---:|"]
    nrun = 0
    for x in out:
        for v, d in sorted(x["method"].items()):
            if v == "none":
                continue
            if d["run"] == 0:
                md.append(f"| {x['key']} | {v} | {d['total']} |")
                nrun += 1
    md.append(f"\n{nrun} (component, method) pairs have zero real-consumer runs.")
    open(f"{OUT}/VARIANTS.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
