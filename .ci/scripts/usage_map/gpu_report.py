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

"""gpu_report.py -> out-cov2/GPU.md, out-cov2/gpu.json

Host-side coverage of the GPU (DPC++) build from sklearnex/onedal GPU test runs
(run_gpu_tests.sh, collect_gpu.py). Covers source files that exist only in the
GPU build, i.e. code the CPU map (report2.py) cannot see. Device kernels are not
instrumented: a kernel body counts through its host-side launcher only.
"""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usage_consts import U, area  # noqa: E402

UNITS = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{U}/data/gpu/gpu_*.json.gz"))


def load(name):
    with gzip.open(name, "rt") as fh:
        return json.load(fh)


def main():
    cpu = set(load(f"{U}/data/cov2/universe.json.gz")["lines"])
    uni = load(f"{U}/data/gpu/universe.json.gz")
    hit = defaultdict(set)
    for u in UNITS:
        for f, ls in load(f"{U}/data/gpu/{u}.json.gz")["lines"].items():
            hit[f] |= set(ls)
    comps = {}
    for f, ls in uni["lines"].items():
        if f in cpu:
            continue
        api, comp, _ = area(f)
        c = comps.setdefault(f"{api}/{comp}", dict(api=api, comp=comp, files=0, lines=0, hit=0))
        c["files"] += 1
        c["lines"] += len(ls)
        c["hit"] += len(hit.get(f, set()) & set(ls))
    rows = sorted(comps.values(), key=lambda c: -(c["lines"] - c["hit"]))
    json.dump(rows, open(f"{U}/out-cov2/gpu.json", "w"), indent=1)
    L = sum(c["lines"] for c in rows)
    H = sum(c["hit"] for c in rows)
    md = ["# GPU-only code (files absent from the CPU build), host-side coverage\n",
          f"{L} lines in {sum(c['files'] for c in rows)} files; GPU tests of sklearnex/onedal run {H} "
          f"({100 * H / max(L, 1):.1f}%), {L - H} never run.\n",
          "| component | api | files | lines | run | never run |", "|---|---|---:|---:|---:|---:|"]
    md += [f"| {c['comp']} | {c['api']} | {c['files']} | {c['lines']} | {c['hit']} | {c['lines'] - c['hit']} |"
           for c in rows]
    open(f"{U}/out-cov2/GPU.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[:2]))
    print("\n".join(md[4:16]))


if __name__ == "__main__":
    main()
