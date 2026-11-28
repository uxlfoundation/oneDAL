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

"""collect.py <build> <unit-dir|universe> [...]  -> data/<build>/<name>.json.gz

Runs gcov over every .gcno of oneDAL-<build> and records, per function, the
execution count one test unit produced (gcda tree cov-<build>/u/<unit>).
'universe' runs with no .gcda: every compiled function, count 0.
Headers seen through the __release_lnx_gnu/include copies are mapped back to
their cpp/ source so header code is attributed to its component.
"""
import concurrent.futures as cf
import gzip
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
B = sys.argv[1]
SRC = f"{U}/oneDAL-{B}"
WORKS = [d for d in (f"{SRC}/__work_gnu", f"{SRC}/__work") if os.path.isdir(d)]
GCOV = os.environ.get("GCOV", "gcov").split()
REL_INC = [f"{SRC}/__release_lnx_gnu/daal/latest/include/", f"{SRC}/__release_lnx/daal/latest/include/"]


def canon(path):
    p = os.path.normpath(os.path.join(SRC, path))
    for inc in REL_INC:
        if p.startswith(inc):
            r = p[len(inc):]
            p = f"{SRC}/cpp/{r}" if r.startswith("oneapi/") else f"{SRC}/cpp/daal/include/{r}"
    if not p.startswith(SRC + "/cpp/"):
        return None
    return p[len(SRC) + 1:]


def gcov_one(gcno, gcda_root):
    obj = gcno[:-5]
    with tempfile.TemporaryDirectory(dir=f"{U}/gcov-work") as td:
        base = os.path.join(td, "o")
        os.symlink(gcno, base + ".gcno")
        if gcda_root:
            gcda = f"{gcda_root}{obj}.gcda"
            if not os.path.exists(gcda):
                return None
            os.symlink(gcda, base + ".gcda")
        r = subprocess.run(GCOV + ["--json-format", "--stdout", "--demangled-names", "-o", td, base],
                           cwd=SRC, capture_output=True, text=True)
    return [json.loads(ln) for ln in r.stdout.splitlines() if ln.strip().startswith("{")]


def collect(name, gcnos):
    gcda_root = None if name == "universe" else f"{U}/cov-{B}/u/{name}"
    funcs = {}                   # mangled -> [file, line, demangled, count]
    lines = defaultdict(dict)    # file -> {line: count}
    loaded = 0
    with cf.ThreadPoolExecutor(32) as ex:
        for docs in ex.map(lambda g: gcov_one(g, gcda_root), gcnos):
            if docs is None:
                continue
            loaded += 1
            for doc in docs:
                for f in doc.get("files", []):
                    fn = canon(f["file"])
                    if fn is None:
                        continue
                    for fu in f.get("functions", []):
                        e = funcs.setdefault(fu["name"], [fn, fu["start_line"], fu.get("demangled_name", fu["name"]), 0])
                        e[3] += fu["execution_count"]
                    lm = lines[fn]
                    for ln in f.get("lines", []):
                        lm[ln["line_number"]] = lm.get(ln["line_number"], 0) + ln["count"]
    os.makedirs(f"{U}/data/{B}", exist_ok=True)
    if name == "universe":
        fl = {k: v[:3] for k, v in funcs.items()}
        ll = {f: sorted(m) for f, m in lines.items()}
    else:  # only what ran: keeps per-unit files small
        fl = {k: v[3] for k, v in funcs.items() if v[3]}
        ll = {f: sorted(n for n, c in m.items() if c) for f, m in lines.items()}
        ll = {f: v for f, v in ll.items() if v}
    with gzip.open(f"{U}/data/{B}/{name}.json.gz", "wt") as fh:
        json.dump({"name": name, "loaded": loaded, "functions": fl, "lines": ll}, fh)
    print(f"{B}/{name}: loaded={loaded} funcs={len(fl)} files={len(ll)}", flush=True)


def main():
    os.makedirs(f"{U}/gcov-work", exist_ok=True)
    gcnos = []
    for w in WORKS:
        for d, _, fs in os.walk(w):
            gcnos += [os.path.join(d, f) for f in fs if f.endswith(".gcno") and "_static" not in d]
    names = sys.argv[2:]
    if names == ["all"]:
        names = ["universe"] + sorted(os.listdir(f"{U}/cov-{B}/u"))
    for n in names:
        collect(n, gcnos)


if __name__ == "__main__":
    main()
