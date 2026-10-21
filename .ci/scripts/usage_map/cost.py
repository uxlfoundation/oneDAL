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

"""cost.py <build> -> data/cost-<build>.json

Per translation unit of a release-flag build: source file, compile seconds
(from the tcxx wrapper log), object .text+.data size, and which library it
lands in. Aggregated later by component in report.py.
"""
import glob
import json
import os
import re
import subprocess
import sys

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
B = sys.argv[1]
SRC = f"{U}/oneDAL-{B}"


def main():
    secs = {}
    for line in open(f"{U}/compile-{B}.log"):
        t, out = line.split(maxsplit=1)
        secs[os.path.normpath(os.path.join(SRC, out.strip()))] = float(t)
    tus = []
    trig = re.compile(r"\.trigger = .* -o(\S+) (\S+\.(?:cpp|cc|c))'?\s*$")
    for d in glob.glob(f"{SRC}/__work_gnu/**/*.o.d", recursive=True):
        txt = open(d, errors="replace").read()
        m = None
        for line in txt.splitlines():
            if ".trigger =" in line:
                m = trig.search(line)
        if not m:
            continue
        obj = os.path.normpath(os.path.join(SRC, m.group(1)))
        if not os.path.exists(obj):
            continue
        tus.append(dict(obj=os.path.relpath(obj, SRC), src=m.group(2), sec=secs.get(obj, 0.0),
                        lib=obj.split("/__work_gnu/md/lnx32e/")[1].split("/")[0]))
    # sizes in one size(1) call per 500 objects
    for i in range(0, len(tus), 500):
        chunk = tus[i:i + 500]
        r = subprocess.run(["size", "-A", "-d"] + [os.path.join(SRC, t["obj"]) for t in chunk],
                           capture_output=True, text=True)
        cur = None
        sizes = {}
        for line in r.stdout.splitlines():
            if line.endswith(":") and "section" not in line:
                cur = line.split(" ")[0]
                sizes[cur] = 0
            parts = line.split()
            if cur and len(parts) == 3 and parts[0].startswith((".text", ".data", ".rodata")):
                sizes[cur] += int(parts[1])
        for t in chunk:
            t["bytes"] = sizes.get(os.path.join(SRC, t["obj"]), 0)
    json.dump(tus, open(f"{U}/data/cost-{B}.json", "w"))
    print(B, "tus", len(tus), "cpu-s", round(sum(t["sec"] for t in tus)),
          "MB", round(sum(t["bytes"] for t in tus) / 1e6, 1))


if __name__ == "__main__":
    main()
