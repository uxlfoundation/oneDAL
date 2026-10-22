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

"""run_bench.py [filter]

Runs every scikit-learn_bench case (sklbench-cases/<case>.json, sklearnex on CPU)
against the instrumented oneDAL in oneDAL-cov2, one unit per case, each with its
own GCOV_PREFIX (cov-cov2/u/bench_<case>). Same PRELOAD/env as run_units.py.
Writes records in the units-cov2.json shape to units-bench.json.
"""
import concurrent.futures as cf
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
B = "cov2"
FILT = sys.argv[1] if len(sys.argv) > 1 else ""
PY = f"{U}/venv/bin/python"
BENCH = f"{U}/sklbench"
CASES = f"{U}/sklbench-cases"
REL = f"{U}/oneDAL-{B}/__release_lnx_gnu/daal/latest"
LIB = f"{REL}/lib/intel64"
TBB = f"{U}/oneDAL-{B}/__deps/tbb/lnx/lib"
PRELOAD = ":".join([f"{TBB}/libtbb.so.12", f"{TBB}/libtbbmalloc.so.2"] +
                   [f"{LIB}/libonedal{x}.so.4" for x in ("_core", "_thread", "", "_parameters")])


def units():
    out = []
    for f in sorted(glob.glob(f"{CASES}/*_float*.json")):
        case = os.path.basename(f)[:-5]
        out.append(dict(id=f"bench:{case}", consumer="workloads", kind="bench", path=f, case=case))
    return [u for u in out if FILT in u["id"]]


def run(u):
    uid = "bench_" + re.sub(r"[^A-Za-z0-9_]", "_", u["case"])
    pre = f"{U}/cov-{B}/u/{uid}"
    shutil.rmtree(pre, ignore_errors=True)
    os.makedirs(pre)
    wd = f"{U}/run-bench/{uid}"
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    res_file = f"{wd}/result.json"
    env = dict(os.environ, GCOV_PREFIX=pre, GCOV_PREFIX_STRIP="0", OMP_NUM_THREADS="4",
               TBB_NUM_THREADS="4", PYTHONHASHSEED="0", SCIPY_ARRAY_API="1",
               LD_PRELOAD=PRELOAD, PATH=f"{U}/venv/bin:" + os.environ.get("PATH", ""))
    cmd = [PY, "-m", "sklbench", "--config", u["path"], "--result-file", res_file, "--runner-log-level", "ERROR"]
    t = time.time()
    r = subprocess.run(cmd, cwd=BENCH, env=env, capture_output=True, text=True, timeout=1800)
    dt = time.time() - t
    summary = ""
    n = 0
    if os.path.exists(res_file):
        cases = json.load(open(res_file)).get("bench_cases", [])
        n = len(cases)
        times = [str(c.get("time[ms]")) for c in cases if c.get("time[ms]") is not None]
        summary = f"bench_cases={n} time_ms=" + ",".join(times)
    if r.returncode != 0 or n == 0:
        tail = (r.stdout + r.stderr).strip().splitlines()
        summary = (tail[-1] if tail else "no output") if n == 0 else summary
        rc = r.returncode or 1
    else:
        rc = 0
    return dict(id=u["id"], uid=uid, consumer=u["consumer"], kind=u["kind"], path=u["path"],
                seconds=round(dt, 2), rc=rc, summary=summary[-200:])


def main():
    us = units()
    path = f"{U}/units-bench.json"
    done = {}
    us.sort(key=lambda u: u["id"])
    with cf.ThreadPoolExecutor(4) as ex:
        for res in ex.map(run, us):
            done[res["id"]] = res
            print(f'{res["seconds"]:8.1f}s rc={res["rc"]} {res["id"]} | {res["summary"][:90]}', flush=True)
            json.dump(list(done.values()), open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
