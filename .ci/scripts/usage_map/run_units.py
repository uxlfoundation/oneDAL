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

"""run_units.py <build> [filter]

Runs every test unit (one pytest file, or one C++ example) against the
instrumented oneDAL in oneDAL-<build>, each with its own GCOV_PREFIX
(cov-<build>/u/<unit-id>), and records wall time + outcome in units-<build>.json.
Per-unit data lets us attribute code to tests and cost to components.
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
B = sys.argv[1]
FILT = sys.argv[2] if len(sys.argv) > 2 else ""
PY = f"{U}/venv/bin/python"
SRC = f"{U}/sklearnex-2026.1"
SP = f"{U}/venv/lib/python3.12/site-packages"
REL = f"{U}/oneDAL-{B}/__release_lnx_gnu/daal/latest"
LIB = f"{REL}/lib/intel64"
TBB = f"{U}/oneDAL-{B}/__deps/tbb/lnx/lib"
PRELOAD = ":".join([f"{TBB}/libtbb.so.12", f"{TBB}/libtbbmalloc.so.2"] +
                   [f"{LIB}/libonedal{x}.so.4" for x in ("_core", "_thread", "", "_parameters")])


def run_capped(cmd, timeout, **kw):
    """subprocess.run that turns a timeout into rc=-9, so one slow unit does not stop the run."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(cmd, -9, out, f"timeout after {timeout} s")


def units():
    out = []
    for pkg, consumer in (("sklearnex", "sklearnex"), ("onedal", "sklearnex"), ("daal4py", "daal4py")):
        for f in sorted(glob.glob(f"{SP}/{pkg}/**/test_*.py", recursive=True)):
            out.append(dict(id=f"py:{pkg}/{os.path.relpath(f, f'{SP}/{pkg}')}", consumer=consumer,
                            kind="pytest", path=f))
    repo = {"test_model_builders.py": "model-builders", "test_daal4py_examples.py": "daal4py-api",
            "test_daal4py_serialization.py": "daal4py-api", "test_npy.py": "daal4py-api"}
    for name, consumer in repo.items():
        out.append(dict(id=f"py:tests/{name}", consumer=consumer, kind="pytest", path=f"{SRC}/tests/{name}"))
    # distributed mode under 4 MPI ranks (impi-rt + mpi4py in the venv)
    out.append(dict(id="mpi:tests/test_daal4py_spmd_examples.py", consumer="daal4py-api", kind="mpi",
                    args=["-s", f"{SRC}/tests/test_daal4py_spmd_examples.py"]))
    # sklearnex/onedal spmd tests need a GPU queue: on CPU all 2113 skip
    for k in ("oneapi", "daal"):
        for e in sorted(glob.glob(f"{REL}/examples/{k}/cpp/_cmake_results/intel_intel64_so/*")):
            out.append(dict(id=f"ex:{k}/{os.path.basename(e)}", consumer=f"examples-{k}", kind="example",
                            path=e, cwd=f"{REL}/examples/{k}/cpp"))
    return [u for u in out if FILT in u["id"]]


def run(u):
    uid = re.sub(r"[^A-Za-z0-9_.-]", "_", u["id"])
    pre = f"{U}/cov-{B}/u/{uid}"
    shutil.rmtree(pre, ignore_errors=True)
    os.makedirs(pre)
    env = dict(os.environ, GCOV_PREFIX=pre, GCOV_PREFIX_STRIP="0", OMP_NUM_THREADS="2",
               PYTHONHASHSEED="0", SCIPY_ARRAY_API="1", TBB_NUM_THREADS="4")
    if u["kind"] == "pytest":
        env["LD_PRELOAD"] = PRELOAD
        wd = f"{U}/run-u/{uid}"
        os.makedirs(wd, exist_ok=True)
        cmd = [PY, "-m", "pytest", "-q", "-n", "4", "--timeout=3600", "-c", f"{SRC}/setup.cfg", u["path"]]
    elif u["kind"] == "mpi":
        env["LD_PRELOAD"] = PRELOAD
        env["I_MPI_FABRICS"] = "shm"
        wd = f"{U}/run-u/{uid}"
        os.makedirs(wd, exist_ok=True)
        cmd = [f"{U}/venv/bin/mpirun", "-n", "4", PY, f"{SRC}/tests/helper_mpi_tests.py", "pytest", "-q",
               "-c", f"{SRC}/setup.cfg", *u["args"]]
    else:
        env["LD_LIBRARY_PATH"] = f"{LIB}:{TBB}"
        wd = u["cwd"]
        cmd = [u["path"]]
    t = time.time()
    r = run_capped(cmd, 7200, cwd=wd, env=env)
    dt = time.time() - t
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
    return dict(u, uid=uid, seconds=round(dt, 2), rc=r.returncode, summary=tail[0][-200:])


def main():
    us = units()
    path = f"{U}/units-{B}.json"
    done = {x["id"]: x for x in json.load(open(path))} if os.path.exists(path) else {}
    # long pytest units first so the pool drains evenly
    us.sort(key=lambda u: (u["kind"] != "pytest", -done.get(u["id"], {}).get("seconds", 0)))
    with cf.ThreadPoolExecutor(int(os.environ.get("JOBS", "24"))) as ex:
        for res in ex.map(run, us):
            done[res["id"]] = res
            print(f'{res["seconds"]:8.1f}s rc={res["rc"]} {res["id"]} | {res["summary"][:90]}', flush=True)
            json.dump(list(done.values()), open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
