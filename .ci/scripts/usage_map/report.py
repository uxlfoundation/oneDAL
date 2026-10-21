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

"""report.py -> out/{functions,files,algorithms}.csv + out/REPORT.md

Joins per-consumer gcov data (collect.py) into usage classes:
  A  used by the sklearnex product (sklearnex/onedal, daal4py.sklearn, model builders)
  P  used only through daal4py public API (exercised by daal4py examples/tests only)
  C  reached only by oneDAL's own examples (filtered consumer: removable)
  E  compiled, reached by nobody
D marks A-less functions whose symbol is exported from libonedal*.so (ABI-only).
"""
import csv
import gzip
import json
import os
import re
import subprocess
from collections import defaultdict

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
REAL = ["sklearnex", "daal4py", "model-builders"]
PUBLIC = ["daal4py-api"]
# static surfaces (2026.1.0): oneapi algos bound in sklearnex onedal/*.cpp,
# daal dirs reachable from daal4py public names, daal dirs sklearnex python calls
PYBIND = set("basic_statistics chebyshev_distance cosine_distance covariance dbscan decision_forest "
             "finiteness_checker kmeans kmeans_init knn linear_kernel linear_regression logistic_regression "
             "minkowski_distance newton_cg pca polynomial_kernel rbf_kernel sigmoid_kernel svm".split())
D4P = set("cholesky cordistance cosdistance covariance dbscan dtrees elastic_net em engines implicit_als "
          "k_nearest_neighbors kernel_function kmeans lasso_regression linear_model linear_regression "
          "logistic_regression low_order_moments multiclassclassifier normalization objective_function "
          "optimization_solver pca pivoted_qr qr regression ridge_regression svd svm tsne classifier".split())
SKLX_D4P = set("kmeans optimization_solver objective_function linear_regression engines dtrees tsne svm "
               "ridge_regression pca multiclassclassifier logistic_regression lasso_regression kernel_function "
               "k_nearest_neighbors elastic_net dbscan covariance cosdistance cordistance linear_model "
               "regression classifier".split())


def static_tag(api, comp):
    if api == "oneapi":
        return "sklearnex-pybind" if comp in PYBIND else ""
    if comp in SKLX_D4P:
        return "sklearnex-via-d4p"
    return "d4p-public" if comp in D4P else ""
FILTERED = ["examples-oneapi", "examples-daal"]
LIBDIR = f"{U}/oneDAL-cov/__release_lnx_gnu/daal/latest/lib/intel64"


def load(label):
    with gzip.open(f"{U}/data/{label}.json.gz", "rt") as fh:
        return json.load(fh)


def area(path):
    """Map a source path to (api, component, sub)."""
    p = path.split("/")
    if p[1] == "daal":
        if p[2] == "src" and p[3] == "algorithms" and len(p) > 5:
            return "daal", p[4], p[5] if len(p) > 6 else "-"
        if p[2] == "src" and p[3] == "algorithms":
            return "daal", "shared:algorithms-base", "-"
        if p[2] == "include" and p[3] == "algorithms" and len(p) > 5:
            return "daal", p[4], "include"
        return "daal", "shared:" + "/".join(p[2:4]), "-"
    if p[1] == "oneapi":
        if p[3] == "algo" and len(p) > 5:
            return "oneapi", p[4], p[5] if len(p) > 6 else "-"
        return "oneapi", "shared:" + "/".join(p[3:4] if len(p) > 4 else ["root"]), "-"
    return p[1], "shared:other", "-"


def exported():
    names = set()
    for so in os.listdir(LIBDIR):
        if so.endswith(".so.4.0"):
            r = subprocess.run(["nm", "-DC", "--defined-only", os.path.join(LIBDIR, so)],
                               capture_output=True, text=True)
            for line in r.stdout.splitlines():
                parts = line.split(" ", 2)
                if len(parts) == 3 and parts[1] in "TW":
                    names.add(norm(parts[2]))
    return names


def norm(sig):
    return re.sub(r"\s+", "", sig)


def main():
    labels = REAL + PUBLIC + FILTERED
    uni = load("universe")
    data = {l: load(l) for l in labels}
    hits = {l: {(f, n, s): c for f, n, s, c in d["functions"]} for l, d in data.items()}
    exp = exported()
    os.makedirs(f"{U}/out", exist_ok=True)

    rows = []
    for f, n, s, _ in uni["functions"]:
        k = (f, n, s)
        used = {l: hits[l].get(k, 0) for l in labels}
        real = [l for l in REAL if used[l]]
        filt = [l for l in FILTERED if used[l]]
        pub = [l for l in PUBLIC if used[l]]
        cls = "A" if real else ("P" if pub else ("C" if filt else "E"))
        abi = "D" if cls != "A" and norm(n) in exp else ""
        api, comp, sub = area(f)
        rows.append([cls, abi, api, comp, sub, f, s, n, "+".join(real), "+".join(filt)])
    with open(f"{U}/out/functions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "abi_only", "api", "component", "sub", "file", "line", "function",
                    "real_consumers", "filtered_consumers"])
        w.writerows(sorted(rows, key=lambda r: (r[3], r[5], int(r[6]))))

    # line-level per file
    ulines = {f: dict(v) for f, v in uni["lines"].items()}
    llines = {l: {f: {n for n, c in v if c} for f, v in d["lines"].items()} for l, d in data.items()}
    frows = []
    for f, lm in ulines.items():
        tot = len(lm)
        if not tot:
            continue
        real = set().union(*(llines[l].get(f, set()) for l in REAL + PUBLIC))
        filt = set().union(*(llines[l].get(f, set()) for l in FILTERED))
        per = [len(llines[l].get(f, set())) for l in labels]
        api, comp, sub = area(f)
        frows.append([api, comp, sub, f, tot, len(real), len(filt - real), tot - len(real | filt), *per])
    with open(f"{U}/out/files.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["api", "component", "sub", "file", "lines", "real_hit", "filtered_only",
                    "unused", *labels])
        w.writerows(sorted(frows, key=lambda r: (r[1], r[3])))

    # algorithm view: component x api, line + function totals
    agg = defaultdict(lambda: defaultdict(int))
    for r in frows:
        a = agg[(r[1], r[0])]
        a["lines"] += r[4]; a["real"] += r[5]; a["filt"] += r[6]; a["unused"] += r[7]
        for i, l in enumerate(labels):
            a[l] += r[8 + i]
    for r in rows:
        a = agg[(r[3], r[2])]
        a["fn"] += 1; a["fn_" + r[0]] += 1; a["fn_D"] += bool(r[1])
        a["static"] = static_tag(r[2], r[3])
    arows = []
    for (comp, api), a in agg.items():
        arows.append([comp, api, a.get("static", ""), a["lines"], a["real"], a["filt"], a["unused"],
                      a["fn"], a["fn_A"], a["fn_P"], a["fn_C"], a["fn_E"], a["fn_D"],
                      *[a[l] for l in labels]])
    arows.sort(key=lambda r: -r[6])
    hdr = ["component", "api", "static", "lines", "used_hit", "examples_only", "unused",
           "fns", "fn_A", "fn_P", "fn_C", "fn_E", "fn_abi_only", *labels]
    with open(f"{U}/out/algorithms.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(arows)

    tot = defaultdict(int)
    for r in rows:
        tot[r[0]] += 1; tot["D"] += bool(r[1])
    L = sum(r[4] for r in frows); R = sum(r[5] for r in frows)
    Fo = sum(r[6] for r in frows); Un = sum(r[7] for r in frows)
    md = [f"# oneDAL 2026.1.0 usage map (CPU, avx512 build)\n",
          f"Functions: {len(rows)}  A={tot['A']}  P={tot['P']}  C={tot['C']}  E={tot['E']}  (ABI-exported non-A: {tot['D']})\n",
          f"Lines: {L}  real-consumer hit={R} ({100*R/L:.1f}%)  examples-only={Fo}  unused={Un} ({100*Un/L:.1f}%)\n",
          "\n## Components by unused lines\n",
          "| component | api | static | lines | used | examples only | unused | " + " | ".join(labels) + " |",
          "|---|---|---|---:|---:|---:|---:|" + "---:|" * len(labels)]
    for r in arows:
        md.append("| " + " | ".join(str(x) for x in (r[:7] + r[13:])) + " |")
    open(f"{U}/out/REPORT.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[:4]))


if __name__ == "__main__":
    main()
