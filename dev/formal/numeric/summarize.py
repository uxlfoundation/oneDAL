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

"""Splits results.json into supported configurations and configurations that create
empty blocks or empty ranks (outside what get_row_block_count/SPMD produce)."""
import json, re, sys
from collections import defaultdict
rows = json.load(open(sys.argv[1]))
def n_of(d): m = re.search(r"_n(\d+)$", d) or re.match(r"n(\d+)$", d); return int(m.group(1))
def empty_part(r):
    if not r["arm"].startswith("replica"): return False
    B = re.search(r"B=(\w+)", r["cfg"]).group(1); R = int(re.search(r"ranks=(\d+)", r["cfg"]).group(1))
    n = n_of(r["data"])
    for k in range(R):
        nr = n * (k + 1) // R - n * k // R
        if nr == 0: return True
        if B != "auto":
            b = int(B); bs = -(-nr // b)
            if -(-nr // bs) != b: return True
    return False
agg = defaultdict(lambda: [0.0, 0.0, 0, ""]); dom = []
for r in rows:
    if "error" in r: continue
    if empty_part(r): dom.append(r); continue
    a = agg[(r["data"], r["type"], r["arm"])]; a[2] += 1
    if r["mean_rel"] is not None: a[0] = max(a[0], r["mean_rel"])
    if r["var_rel"] is not None and r["var_rel"] >= a[1]: a[1] = r["var_rel"]; a[3] = r["cfg"]
print("| data | type | arm | configs | max rel err mean | max rel err var | worst var config |\n|---|---|---|---|---|---|---|")
for (d, t, arm), (em, ev, c, w) in sorted(agg.items()):
    print(f"| {d} | {t} | {arm} | {c} | {em:.2e} | {ev:.2e} | {w} |")
print(f"\nOutside the supported domain (empty block or empty rank), {len(dom)} runs:")
for r in sorted({(r['data'], r['arm'], re.sub(r' order=\w+', '', r['cfg']), str(r['var'])) for r in dom}):
    print("-", " ".join(r))
