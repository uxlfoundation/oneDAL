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

"""path_attr.py <build> -> out-<build>/PATHS.md + paths.json

For every DAAL row tiered T2s or T2m in out-<build>/CANDIDATES.md, attribute each A line
(executed by a real consumer) to the entry path it is reached through:

  oneapi-sk    reached forward from oneapi/dal/algo/X (call graph; other algo
               entries are not crossed) and X is run by sklearnex on CPU
               -> path (a): sklearnex -> oneapi/dal -> DAAL kernel
  oneapi-nosk  reached from oneapi/dal/algo/X, but no sklearnex CPU unit runs X
  daal4py      no oneapi ancestor; only daal4py / daal4py-api / model-builders run it
               -> path (b)/(c): daal4py wrapper or daal4py estimators -> DAAL public API
  sk-other     sklearnex (or workloads) runs it, no oneapi ancestor in the tree
               (entry lies outside the tree, e.g. daal4py C++ called from python)
  gpu-only     only sklearnex-gpu runs it (GPU build; host DAAL code, not in CPU graph)

Lines: A lines of the CANDIDATES row are re-derived exactly as report2.py does
(union of REAL unit line sets, GPU units folded in) and checked against the row.
Line -> entry category uses the owning function (functions.csv start lines).
Public-API layer (heuristic, reported only): include/ headers and
*types/input/result/parameter/container/dispatcher/iface sources.

Verdict per row, "deprecate daal4py wrapper + DAAL public API, keep kernel":
  confirmed  every A line has an oneapi ancestor that sklearnex runs
  not-safe   some A lines lose their only entry (daal4py / sk-other lines)
  unclear    no line lost, but some have no static entry evidence (gpu-only, oneapi-nosk, other,
             sk-internal, sk-nopath)

Bridges B1-B7 are hand-written sklearnex -> daal4py links, each site re-read and regex-checked.
Live bridges are attached to the rows they cover ("bridges" in paths.json).
Read-only on the workdir (USAGE_MAP_WORK, default cwd); writes only into out-<build>/.
candidates.py reads paths.json: confirmed -> "T2s deprecate API now", the rest -> "T2m migrate sklearnex first".
"""
import csv
import gzip
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usage_consts import CPU, FILTERED, PUBLIC, REAL, U, area, load_units  # noqa: E402

B = sys.argv[1] if len(sys.argv) > 1 else "cov2"
OUT = os.path.join(U, f"out-{B}")
SKX = os.path.join(U, "sklearnex-2026.1")
SK_CPU = {"sklearnex", "workloads"}
CPU_REAL = {"sklearnex", "workloads", "daal4py", "model-builders"}
D4P = {"daal4py", "daal4py-api", "model-builders"}
# DAAL batch-dispatch sources: the public layer that daal4py calls into (not kernels)
PUB_SRC = re.compile(r"^cpp/daal/src/algorithms/[^/]+/[^/]*batch[^/]*\.(cpp|i)$")
PUB_RE = re.compile(r"(^cpp/daal/include/)|(/[^/]*(types|input|result|parameter|container|dispatcher|iface|interface)[^/]*\.(h|cpp|i)$)")


def load(p):
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


def oneapi_algo(f):
    if f and f.startswith("cpp/oneapi/dal/algo/"):
        return f.split("/")[4]
    return None


def load_time_lines(f):
    """Serialization-registration lines (report2.load_time): excluded from GPU hits."""
    try:
        src = open(os.path.join(U, "oneDAL-gpu", f), errors="replace").read().splitlines()
    except OSError:
        return set()
    return {i + 1 for i, t in enumerate(src) if t.lstrip().startswith("__DAAL_REGISTER_SERIALIZATION_CLASS")}


def candidate_rows():
    # T2m is included too: a re-run after candidates.py split the T2s rows must see all of them
    rows = []
    for line in open(os.path.join(OUT, "CANDIDATES.md")):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 7 and c[0].startswith(("T2s", "T2m")):
            rows.append(dict(api=c[1], comp=c[2], sub=c[3], lines=int(c[4]), A=int(c[5]), P=int(c[6])))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- universe, call graph, functions -------------------------------------------------
    uni = load(f"{U}/data/{B}/universe.json.gz")
    fm = uni["functions"]                                   # mangled -> [file, line, demangled]
    ulines = {f: set(v) for f, v in uni["lines"].items()}
    guni = load(f"{U}/data/gpu/universe.json.gz")
    for f, v in guni["lines"].items():                      # report2: GPU lines join the universe
        ulines.setdefault(f, set()).update(v)
    key2m = defaultdict(set)                                # (file, CPU-normalized name) -> mangled
    for m, (f, _, d) in fm.items():
        key2m[(f, CPU.sub("CPU", d))].add(m)
    cg_raw = load(f"{U}/data/{B}/callgraph.json.gz")
    # Node = (file, CPU-normalized demangled name). Mangled names of one function differ between
    # C1/C2 (ctor) and D0/D1/D2 (dtor) aliases; the call graph names the alias that was called, so
    # names absent from the universe are demangled (c++filt) and joined on the normalized name.
    nid_of = {m: (f, CPU.sub("CPU", d)) for m, (f, _, d) in fm.items()}
    dem2nid = defaultdict(set)
    for nid in set(nid_of.values()):
        dem2nid[nid[1]].add(nid)
    missing = sorted({n for a, bs in cg_raw.items() for n in [a, *bs] if n not in nid_of})
    dem = subprocess.run(["c++filt"], input="\n".join(missing), capture_output=True, text=True).stdout.splitlines()
    alias = {}
    for m, d in zip(missing, dem):
        alias[m] = dem2nid.get(CPU.sub("CPU", d), set())
    def nids(n):
        return [nid_of[n]] if n in nid_of else list(alias.get(n, ()))
    cg = defaultdict(set)
    rev = defaultdict(set)
    n_alias_edges = 0
    for a, bs in cg_raw.items():
        for ka in nids(a):
            for b in bs:
                for kb in nids(b):
                    if ka != kb:
                        cg[ka].add(kb)
                        rev[kb].add(ka)
                        n_alias_edges += (b not in nid_of)
    print("callgraph: nodes", len(set(cg) | set(rev)), "edges", sum(map(len, cg.values())), "via alias", n_alias_edges)

    fn = list(csv.DictReader(open(f"{U}/out-{B}/functions.csv")))
    by_file = defaultdict(list)                             # owner ranges per file
    for r in fn:
        r["groups_l"] = r["groups"].split("+") if r["groups"] else []
        r["nid"] = (r["file"], r["function"])
        by_file[r["file"]].append((int(r["line"]), r))
    for f in by_file:
        by_file[f].sort(key=lambda x: x[0])

    # ---- A / P line sets exactly as report2.py ----------------------------------------
    units = {u["uid"]: u for u in load_units(B)}
    gl = {g: defaultdict(set) for g in REAL + PUBLIC + FILTERED}
    for uid, u in units.items():
        p = f"{U}/data/{B}/{uid}.json.gz"
        if os.path.exists(p):
            for f, v in load(p)["lines"].items():
                gl[u["consumer"]][f] |= set(v)
    for name in sorted(os.listdir(f"{U}/data/gpu")):
        if not name.startswith("gpu_"):
            continue
        d = load(f"{U}/data/gpu/{name}")
        for f, v in d["lines"].items():
            gl["sklearnex-gpu"][f] |= set(v) - load_time_lines(f)
    A_line = {}
    P_line = {}
    for f, lines in ulines.items():
        a = set().union(*(gl[g].get(f, set()) for g in REAL)) & lines
        p = set().union(*(gl[g].get(f, set()) for g in PUBLIC)) & lines - a
        A_line[f] = a
        P_line[f] = p

    # ---- forward labels: oneapi algorithm X -> nodes it reaches ----------------------------
    # BFS from X's oneapi functions; never enters another algorithm's oneapi functions, so the
    # label is the entry algorithm. Shared oneapi backend code is passed through.
    label = defaultdict(set)
    srcs = defaultdict(list)
    for nid in list(rev) + list(cg):
        x = oneapi_algo(nid[0])
        if x:
            srcs[x].append(nid)
    for x, s in srcs.items():
        seen = set(s)
        stack = list(s)
        while stack:
            n = stack.pop()
            for c in cg.get(n, ()):
                if c in seen:
                    continue
                y = oneapi_algo(c[0])
                if y is not None and y != x:
                    continue
                seen.add(c)
                stack.append(c)
                label[c].add(x)
    # oneapi algorithms that sklearnex runs on CPU
    x_sk = defaultdict(bool)
    for r in fn:
        x = oneapi_algo(r["file"])
        if x and "sklearnex" in r["groups_l"]:
            x_sk[x] = True
    # direct oneapi callers, for the report
    def direct_oneapi(nid):
        return sorted({oneapi_algo(c[0]) for c in rev.get(nid, ()) if oneapi_algo(c[0])})

    # ---- classify every A function by entry path -----------------------------------------
    def cat_of(r):
        g = set(r["groups_l"])
        if "sklearnex-gpu" in g and not (g & CPU_REAL):
            return "gpu-only"
        labs = label.get(r["nid"], set())
        if {x for x in labs if x_sk[x]}:
            return "oneapi-sk"
        if labs:
            return "oneapi-nosk"
        if g & D4P:
            return "daal4py"
        if g & SK_CPU:
            return "sk-other"
        return "other"

    for r in fn:
        r["cat"] = cat_of(r) if r["class"] == "A" else None

    def owner(f, ln):
        best = None
        for start, r in by_file.get(f, []):
            if start <= ln:
                best = r
            else:
                break
        return best

    rows = candidate_rows()
    # ---- static reachability from sklearnex-run code (bridge evidence) ----------------------
    # reach_all: reachable from any function run by a sklearnex CPU unit.
    # reach_np : reachable from such a function without entering DAAL public-layer files.
    # sk-public: reachable from sklearnex-run code only through the public layer (every static
    # path enters it). sk-internal: some path avoids it. sk-nopath: no static path (virtual,
    # function-pointer and Python-side calls are invisible to the direct-call graph).
    sk_nodes = [r["nid"] for r in fn if set(r["groups_l"]) & SK_CPU]

    def bfs(srcs, block_pub):
        seen = set(srcs)
        q = list(srcs)
        i = 0
        while i < len(q):
            n = q[i]
            i += 1
            for c in cg.get(n, ()):
                if c in seen or (block_pub and (PUB_RE.search(c[0]) or PUB_SRC.search(c[0]))):
                    continue
                seen.add(c)
                q.append(c)
        return seen

    def reach_for(key):
        # sources: sklearnex-run functions of another component (not the row's component), not in the
        # oneapi layer. A same-component caller is part of the kernel family, not an entry into it.
        outside = [n for n in sk_nodes if area(n[0]) != key and oneapi_algo(n[0]) is None]
        return dict(all=bfs(outside, False),
                    np=bfs([n for n in outside if not PUB_RE.search(n[0]) and not PUB_SRC.search(n[0])],
                           True))

    def bridge_of(nid, R):
        if nid in R["np"]:
            return "sk-internal"
        if nid in R["all"]:
            return "sk-public"
        return "sk-nopath"

    # ---- per-unit line hits on A lines (which units execute which category) ----------------
    unit_meta = {}                                          # label -> consumer
    unit_hits = {}                                          # label -> set of (file, line)
    for uid, u in units.items():
        p = f"{U}/data/{B}/{uid}.json.gz"
        if not os.path.exists(p):
            continue
        lab = u.get("id") or uid
        unit_meta[lab] = u["consumer"]
        unit_hits[lab] = {(f, ln) for f, v in load(p)["lines"].items() if f in A_line
                          for ln in v if ln in A_line[f]}
    _tl = {}

    def tlines(f):
        if f not in _tl:
            _tl[f] = load_time_lines(f)
        return _tl[f]

    for name in sorted(os.listdir(f"{U}/data/gpu")):
        if not name.startswith("gpu_"):
            continue
        d = load(f"{U}/data/gpu/{name}")
        lab = name[: -len(".json.gz")] if name.endswith(".json.gz") else name
        unit_meta[lab] = "sklearnex-gpu"
        unit_hits[lab] = {(f, ln) for f, v in d["lines"].items() if f in A_line
                          for ln in v if ln in A_line[f] and ln not in tlines(f)}

    out = []
    for t in rows:
        key = (t["api"], t["comp"], t["sub"])
        sel = [r for r in fn if r["class"] == "A" and (r["api"], r["component"], r["sub"]) == key]
        fcounts = defaultdict(int)
        for r in sel:
            fcounts[r["cat"]] += 1
        A_rows = P_rows = 0
        lc = defaultdict(int)
        ent = defaultdict(int)
        ent_d4p = defaultdict(int)
        pub_lines = defaultdict(int)
        bridge = defaultdict(lambda: defaultdict(int))      # category -> bridge status -> lines
        R = reach_for(key)
        cat_lines = defaultdict(set)                        # category -> {(file, line)}
        owners = defaultdict(lambda: defaultdict(int))      # category -> owner function -> lines
        for f in ulines:
            if area(f) != key:
                continue
            A_rows += len(A_line[f])
            P_rows += len(P_line[f])
            for ln in A_line[f]:
                # which REAL groups executed this exact line (per-unit gcov line hits)
                hitg = {g for g in REAL if ln in gl[g].get(f, ())}
                o = owner(f, ln)
                labs = label.get((f, o["function"]), set()) if o else set()
                osk = {x for x in labs if x_sk[x]}
                sk_hit = bool(hitg & {"sklearnex", "workloads"})
                d4p_hit = bool(hitg & {"daal4py", "model-builders"})
                if o is None:
                    c = "unowned"
                elif sk_hit and osk:
                    c = "oneapi-sk"
                elif sk_hit and d4p_hit:
                    c = "sk+daal4py-no-oneapi"
                elif sk_hit:
                    c = "sk-other"
                elif d4p_hit:
                    c = "daal4py"
                elif "sklearnex-gpu" in hitg:
                    c = "gpu-only"
                else:
                    c = "other"
                lc[c] += 1
                cat_lines[c].add((f, ln))
                if o is not None:
                    owners[c][o["function"][:110]] += 1
                if c in ("sk-other", "sk+daal4py-no-oneapi"):
                    bridge[c][bridge_of(o["nid"], R)] += 1
                if c == "oneapi-sk":
                    for x in sorted(osk):
                        ent[f"{x}"] += 1
                if d4p_hit and not sk_hit:
                    ent_d4p["+".join(sorted(hitg & {"daal4py", "model-builders"}))] += 1
                if PUB_RE.search(f):
                    pub_lines[c] += 1
        units_by_cat = {}
        for c, ls in cat_lines.items():
            per = defaultdict(set)
            for lab, hs in unit_hits.items():
                if hs & ls:
                    per[unit_meta[lab]].add(lab)
            units_by_cat[c] = {k: len(v) for k, v in sorted(per.items())}
        out.append(dict(t, A_check=A_rows, P_check=P_rows, fns=len(sel), fn_cats=dict(fcounts),
                        line_cats=dict(lc), pub_line_cats=dict(pub_lines),
                        oneapi_entries=dict(sorted(((k, v) for k, v in ent.items() if v),
                                                   key=lambda kv: -kv[1])[:8]),
                        daal4py_groups=dict(ent_d4p),
                        bridge={c: dict(v) for c, v in bridge.items()},
                        units_by_cat=units_by_cat,
                        owner_examples={c: sorted(v.items(), key=lambda kv: -kv[1])[:3]
                                        for c, v in owners.items()}))

    for o in out:
        lc = o["line_cats"]
        br = o["bridge"]
        sk_pub = sum(br.get(c, {}).get("sk-public", 0) for c in ("sk-other", "sk+daal4py-no-oneapi"))
        sk_int = sum(br.get(c, {}).get("sk-internal", 0) + br.get(c, {}).get("sk-nopath", 0)
                     for c in ("sk-other", "sk+daal4py-no-oneapi"))
        o["lost_breakdown"] = {"daal4py-only": lc.get("daal4py", 0), "sk-public": sk_pub}
        o["lost_lines"] = lc.get("daal4py", 0) + sk_pub
        o["unknown_lines"] = sum(lc.get(k, 0) for k in ("gpu-only", "unowned", "other", "oneapi-nosk")) + sk_int
        o["kept_lines"] = lc.get("oneapi-sk", 0)
        o["verdict"] = ("not-safe" if o["lost_lines"] else "unclear" if o["unknown_lines"] else "confirmed")
        o["A_match"] = o["A_check"] == o["A"]
        o["P_match"] = o["P_check"] == o["P"]
    res = check_bridges()
    for o in out:
        keys = {o["comp"], f"{o['comp']}/{o['sub']}"}
        o["bridges"] = [b["id"].split()[0] for b in res
                        if b["status"].startswith("live") and b["check_ok"] and keys & set(b["keys"])]
    json.dump({"rows": out, "bridges": res}, open(os.path.join(OUT, "paths.json"), "w"), indent=1)
    write_md(out)
    write_bridges(res)
    for o in out:
        print(o["comp"], o["sub"], "A", o["A"], o["A_check"], "P", o["P"], o["P_check"],
              dict(o["line_cats"]), o["verdict"])


# Hand-written daal4py bridges. Each site is re-read and its regex checked by check_bridges();
# DAAL headers are checked for existence under oneDAL-rel. The daal4py C++ sources are not in the
# checkout, so the daal4py function -> DAAL component link is the Python call site plus the DAAL
# header that carries the same algorithm name. keys: rows (comp or comp/sub) a live bridge covers.
BRIDGES = [
    dict(id="B1 sklearn.metrics.pairwise_distances (cosine, correlation)",
         sk=[("sklearnex/metrics/pairwise.py", 17, r"import pairwise_distances")],
         sk_dispatch=("sklearnex/dispatcher.py", 269, r"sklearn\.metrics\.pairwise_distances"),
         d4p=[("daal4py/sklearn/metrics/_pairwise.py", 67, r"daal4py\.cosine_distance"),
              ("daal4py/sklearn/metrics/_pairwise.py", 74, r"daal4py\.correlation_distance"),
              ("daal4py/sklearn/metrics/_pairwise.py", 139, r"return _daal4py_cosine_distance_dense")],
         daal=["cpp/daal/include/algorithms/distance/cosine_distance.h",
               "cpp/daal/include/algorithms/distance/correlation_distance.h"],
         keys=["cordistance", "cosdistance"],
         rows="cordistance, cosdistance", status="live",
         note="Gate in _pairwise_distances (_pairwise.py:83-141): metric in {cosine, correlation}, Y is None, "
              "dense float64 input; otherwise sklearn fallback."),
    dict(id="B2 ElasticNet / Lasso (sklearnex re-exports the daal4py classes)",
         sk=[("sklearnex/linear_model/coordinate_descent.py", 18,
              r"from daal4py\.sklearn\.linear_model import ElasticNet, Lasso")],
         d4p=[("daal4py/sklearn/linear_model/_coordinate_descent.py", 190, r"daal4py\.optimization_solver_mse"),
              ("daal4py/sklearn/linear_model/_coordinate_descent.py", 195,
               r"daal4py\.optimization_solver_coordinate_descent"),
              ("daal4py/sklearn/linear_model/_coordinate_descent.py", 218, r"daal4py\.elastic_net_training"),
              ("daal4py/sklearn/linear_model/_coordinate_descent.py", 385, r"daal4py\.lasso_regression_training")],
         daal=["cpp/daal/include/algorithms/elastic_net/elastic_net_training_batch.h",
               "cpp/daal/include/algorithms/lasso_regression/lasso_regression_training_batch.h",
               "cpp/daal/include/algorithms/optimization_solver/coordinate_descent/coordinate_descent_batch.h",
               "cpp/daal/include/algorithms/optimization_solver/objective_function/mse_batch.h"],
         keys=["elastic_net", "lasso_regression", "optimization_solver/coordinate_descent",
               "optimization_solver/include"],
         rows="elastic_net, lasso_regression, optimization_solver (coordinate_descent, include)", status="live",
         note="_coordinate_descent.py:584-590: when the daal4py fit returns None the sklearn fit runs instead."),
    dict(id="B3 LogisticRegressionCV (sklearnex.preview, opt-in)",
         sk=[("sklearnex/preview/linear_model/logistic_regression.py", 36,
              r"fit = _daal4py_LogisticRegressionCV\.fit")],
         d4p=[("daal4py/sklearn/linear_model/logistic_path.py", 477, r"def daal4py_fit_cv"),
              ("daal4py/sklearn/linear_model/logistic_path.py", 897, r"def logistic_regression_path_dispatcher"),
              ("daal4py/sklearn/linear_model/logistic_path.py", 1027, r"logistic_regression_path_d4p\("),
              ("daal4py/sklearn/linear_model/logistic_path.py", 311, r"func = _daal4py_loss_and_grad")],
         daal=["cpp/daal/include/algorithms/optimization_solver/objective_function/logistic_loss_batch.h",
               "cpp/daal/include/algorithms/optimization_solver/objective_function/cross_entropy_loss_batch.h"],
         keys=["logistic_regression/include", "objective_function/logistic_loss",
               "objective_function/cross_entropy_loss"],
         rows="logistic_regression (include), objective_function (logistic_loss, cross_entropy_loss)",
         status="live (guard daal >= 2024.1, sklearnex/preview/linear_model/logistic_regression.py:19)",
         note="daal4py_fit_cv:477 patches _logistic_regression_path and runs the sklearn fit; the patched path "
              "reaches logistic_regression_path_dispatcher:897, which calls logistic_regression_path_d4p (:1027)."),
    dict(id="B4 LogisticRegression (non-preview)",
         sk=[("sklearnex/linear_model/logistic_regression.py", 26, r"if daal_check_version\(\(2024, \"P\", 1\)\)")],
         d4p=[("sklearnex/linear_model/logistic_regression.py", 612,
               r"LogisticRegression = _daal4py_LogisticRegression")],
         daal=[], keys=["logistic_regression"], rows="logistic_regression -", status="guarded-out",
         note="daal4py LogisticRegression is assigned only in the else branch (line 611) of the daal >= 2024.1 check."),
    dict(id="B5 Ridge / KMeans / PCA fallbacks",
         sk=[("sklearnex/linear_model/ridge.py", 21, r"if daal_check_version\(\(2024, \"P\", 600\)\)"),
             ("sklearnex/cluster/k_means.py", 21, r"if daal_check_version\(\(2023, \"P\", 200\)\)"),
             ("sklearnex/decomposition/pca.py", 21, r"if daal_check_version\(\(2024, \"P\", 100\)\)")],
         d4p=[("sklearnex/linear_model/ridge.py", 417, r"from daal4py\.sklearn\.linear_model import Ridge"),
              ("sklearnex/cluster/k_means.py", 552, r"from daal4py\.sklearn\.cluster import KMeans"),
              ("sklearnex/decomposition/pca.py", 555, r"from daal4py\.sklearn\.decomposition import PCA")],
         daal=[], keys=["ridge_regression", "kmeans", "pca"], rows="ridge_regression, kmeans, pca",
         status="guarded-out",
         note="daal4py imports sit in else branches of the version guards (ridge.py:416, k_means.py:551, pca.py:554)."),
    dict(id="B6 SVC (daal4py)",
         sk=[("sklearnex/svm/__init__.py", 19, r"if daal_check_version\(\(2021, \"P\", 300\)\)")],
         d4p=[("sklearnex/svm/__init__.py", 24, r"from daal4py\.sklearn\.svm import SVC")],
         daal=[], keys=["svm"], rows="svm", status="guarded-out", note="Else branch only when daal < 2021.3.0."),
    dict(id="B7 mahalanobis (covariance)",
         sk=[("sklearnex/covariance/incremental_covariance.py", 51,
              r"partial\(pairwise_distances, metric=\"mahalanobis\"\)"),
             ("sklearnex/preview/covariance/covariance.py", 47,
              r"partial\(pairwise_distances, metric=\"mahalanobis\"\)")],
         d4p=[("daal4py/sklearn/metrics/_pairwise.py", 83, r"def _pairwise_distances")],
         daal=[], keys=["covariance"], rows="covariance -, covariance include", status="no DAAL path",
         note="_pairwise_distances gates on metric in {cosine, correlation}; mahalanobis fails the gate and goes to "
              "_sklearn_pairwise_distances. No DAAL kernel is reached through this route."),
]


def check_bridges():
    res = []
    for b in BRIDGES:
        ok = True
        sites = []
        for path, ln, rx in b["sk"] + b["d4p"] + ([b["sk_dispatch"]] if "sk_dispatch" in b else []):
            text = open(os.path.join(SKX, path), encoding="utf-8").read().splitlines()
            # the regex must match the cited line or one within 3 lines of it
            hit = [i + 1 for i in range(max(0, ln - 4), min(len(text), ln + 3)) if re.search(rx, text[i])]
            ok &= bool(hit)
            tag = "ok" if hit else "MISMATCH"
            sites.append(f"{path}:{ln} {tag}" + (f" (matched {hit[0]})" if hit else ""))
        hdr = []
        for h in b["daal"]:
            e = os.path.exists(os.path.join(U, "oneDAL-rel", h))
            ok &= e
            hdr.append(f"{h} {'ok' if e else 'MISSING'}")
        res.append(dict(b, sites=sites, headers=hdr, check_ok=ok))
    return res


def write_bridges(res):
    L = ["", "## daal4py bridges (hand-written; each site re-read and regex-checked by the script)", "",
         "| bridge | sklearnex site | daal4py site(s) | DAAL headers (oneDAL-rel) | rows | status | check |",
         "|---|---|---|---|---|---|---|"]
    for b in res:
        L.append(f"| {b['id']} | {'; '.join(s for s in b['sites'] if s.startswith('sklearnex')) or '-'} | "
                 f"{'; '.join(s for s in b['sites'] if s.startswith('daal4py')) or '-'} | "
                 f"{'; '.join(b['headers']) or '-'} | {b['rows']} | {b['status']} | "
                 f"{'ok' if b['check_ok'] else 'FAIL'} |")
    L += ["", "Notes:", ""] + [f"- {b['id']}: {b['note']}" for b in res]
    with open(os.path.join(OUT, "PATHS.md"), "a") as fh:
        fh.write("\n".join(L) + "\n")


def write_md(out):
    L = [f"# Entry-path attribution, build {B}", "",
         f"Generated by `path_attr.py`. Per DAAL row of `out-{B}/CANDIDATES.md` tiered T2s or T2m, each A line is attributed",
         "to the entry path real consumers reach it through (line -> owning function, functions.csv starts).",
         "`A ok`/`P ok` = the re-derived line counts equal the CANDIDATES row.", "",
         "Paths: **oneapi-sk** = (a) sklearnex -> oneapi/dal algo run by sklearnex -> DAAL kernel; "
         "**daal4py** = only daal4py / daal4py-api / model-builders run it, no oneapi ancestor (path b/c); "
         "**sk+daal4py-no-oneapi** / **sk-other** = sklearnex runs it, no in-tree oneapi ancestor; "
         "**gpu-only** = sklearnex-gpu only; **unowned** = line-to-function attribution failed.", "",
         "Verdict: **not-safe** if an A line is lost when the wrapper goes (daal4py-only, or sklearnex-run and "
         "reachable only through the DAAL public layer); **unclear** if an A line has no static evidence either way "
         "(sk-internal, sk-nopath, gpu-only, other); **confirmed** otherwise (includes rows with 0 A lines).", "",
         "| comp | sub | lines | A | P | A ok | P ok | oneapi-sk | daal4py | sk-public | sk-internal | sk-nopath | gpu-only | other | verdict | bridges |",
         "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for o in out:
        lc = o["line_cats"]
        br = o["bridge"]

        def bsum(st):
            return sum(br.get(c, {}).get(st, 0) for c in ("sk-other", "sk+daal4py-no-oneapi"))
        other = sum(lc.get(k, 0) for k in ("unowned", "other", "oneapi-nosk"))
        L.append(f"| {o['comp']} | {o['sub']} | {o['lines']} | {o['A']} | {o['P']} | "
                 f"{'yes' if o['A_match'] else 'NO'} | {'yes' if o['P_match'] else 'NO'} | "
                 f"{lc.get('oneapi-sk', 0)} | {lc.get('daal4py', 0)} | {bsum('sk-public')} | "
                 f"{bsum('sk-internal')} | {bsum('sk-nopath')} | {lc.get('gpu-only', 0)} | {other} | "
                 f"{o['verdict']} | {', '.join(o['bridges']) or '-'} |")
    vc = Counter(o["verdict"] for o in out)
    L += ["", f"Counts over {len(out)} rows: " + ", ".join(f"{k} {vc[k]}" for k in ("confirmed", "not-safe", "unclear")) + ".", "",
          "Limits and corrections (read before using the verdicts):", "",
          "- Entry path (a) is the only path with a source-verified oneapi route: `oneapi-sk` lines. Where "
          "`oneapi-sk` is 0 and the row is `not-safe`, the wrapper is the only sklearnex route found (rows "
          "cordistance, cosdistance, elastic_net, lasso_regression, optimization_solver coordinate_descent, "
          "logistic_regression include via preview CV). Those verdicts are backed by bridges B1-B3, not only by the graph.",
          "- The graph is direct calls only. DAAL dispatches through virtual `ContainerIface`/`Batch` objects and "
          "daal4py is Python-side, so `sk-nopath` lines are unresolved, not proof of no sklearnex path. Those lines "
          "make a row `unclear` unless a daal4py-only line already makes it `not-safe`.",
          "- `sk-internal` / `sk-public` use sklearnex-run functions of other components as sources. Same-component "
          "callers are excluded so a kernel does not count as its own entry.",
          "- Earlier claim retracted: covariance mahalanobis is NOT a DAAL path. `_pairwise_distances` gates on "
          "cosine/correlation and falls back to sklearn for mahalanobis (bridge B7).",
          "- B4, B5, B6 (LogisticRegression, Ridge, KMeans, PCA, SVC daal4py fallbacks) are guarded out for daal "
          "2026.1 and contribute no lines.", ""]
    L += ["## Evidence per row", ""]
    for o in out:
        L.append(f"### {o['comp']} / {o['sub']} (A={o['A']}, verdict {o['verdict']})")
        L.append("")
        L.append(f"- lost: daal4py-only {o['lost_breakdown']['daal4py-only']}, sk-public "
                 f"{o['lost_breakdown']['sk-public']}; kept (oneapi-sk) {o['kept_lines']}; "
                 f"unknown {o['unknown_lines']}")
        for c in ("oneapi-sk", "daal4py", "sk+daal4py-no-oneapi", "sk-other", "gpu-only", "other", "unowned"):
            if o["line_cats"].get(c):
                ex = "; ".join(f"`{n}` ({k})" for n, k in o["owner_examples"].get(c, [])) or "-"
                un = ", ".join(f"{k} {v}" for k, v in o["units_by_cat"].get(c, {}).items()) or "-"
                L.append(f"- {c}: {o['line_cats'][c]} lines; units [{un}]; owners: {ex}")
        if o["oneapi_entries"]:
            L.append(f"- oneapi entry algos: {o['oneapi_entries']}")
        L.append("")
    open(os.path.join(OUT, "PATHS.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
