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

"""callgraph.py <build> -> data/<build>/callgraph.json.gz

Reads gcc -fcallgraph-info (.ci, VCG) of every dynamic-lib TU and merges them
into one graph keyed by mangled name (TU prefix dropped), so calls across TUs
link up. Only direct calls: virtual / function-pointer calls are absent.
"""
import concurrent.futures as cf
import gzip
import json
import os
import re
import sys
from collections import defaultdict

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
B = sys.argv[1]
SRC = f"{U}/oneDAL-{B}"
EDGE = re.compile(r'^edge: \{ sourcename: "[^"]*?:([^":]+)" targetname: "(?:[^"]*?:)?([^":]+)"')


def parse(path):
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = EDGE.match(line)
            if m and m.group(1) != m.group(2):
                out.append((m.group(1), m.group(2)))
    return out


def main():
    cis = []
    for d, _, fs in os.walk(f"{SRC}/__work_gnu"):
        if "_static" not in d:
            cis += [os.path.join(d, f) for f in fs if f.endswith(".ci")]
    g = defaultdict(set)
    with cf.ThreadPoolExecutor(32) as ex:
        for edges in ex.map(parse, cis):
            for a, b in edges:
                g[a].add(b)
    os.makedirs(f"{U}/data/{B}", exist_ok=True)
    with gzip.open(f"{U}/data/{B}/callgraph.json.gz", "wt") as fh:
        json.dump({k: sorted(v) for k, v in g.items()}, fh)
    print(B, "ci", len(cis), "callers", len(g), "edges", sum(map(len, g.values())))


if __name__ == "__main__":
    main()
