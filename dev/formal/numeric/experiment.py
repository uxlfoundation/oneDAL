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

"""Numeric experiment for the count/mean/M2 merge. See README.md in dev/formal.

Arms: (1) host replica of the GPU kernel arithmetic (replica.cpp), (2) real oneDAL CPU
via daal4py low_order_moments (batch and streaming with different chunkings).
Reference: exact rational arithmetic (fractions) on the exact binary values fed to each arm.
"""
import itertools, json, subprocess, sys, tempfile
from fractions import Fraction
import numpy as np

HERE = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(20261111)

def datasets():
    yield "normal_0_1_n20000", rng.normal(0, 1, 20000)
    yield "offset1e4_sd1e-2_n20000", 1e4 + rng.normal(0, 1e-2, 20000)
    yield "offset1e8_sd1e-3_n20000", 1e8 + rng.normal(0, 1e-3, 20000)
    yield "offset1e8_sd1e-3_n150000", 1e8 + rng.normal(0, 1e-3, 150000)
    yield "constant_3.1_n20000", np.full(20000, 3.1)
    yield "n2", np.array([1.0, 2.0])
    yield "n1", np.array([5.0])

def exact(x):
    q = [Fraction(float(v)) for v in x]
    n = len(q); m = sum(q) / n
    m2 = sum((v - m) ** 2 for v in q)
    var = m2 / (n - 1) if n > 1 else None
    return m, var

def relerr(got, ref):
    if ref is None: return None
    if not np.isfinite(got): return float("inf")
    if ref == 0: return abs(got)
    return float(abs(Fraction(got) - ref) / abs(ref))

def replica(binpath, exe, ftype, B, L, order, seed, R):
    out = subprocess.run([exe, binpath, ftype, str(B), str(L), order, str(seed), str(R)],
                         capture_output=True, text=True, check=True).stdout.split()
    return float(out[0]), float(out[1])

def daal(x, fptype, chunks):
    import daal4py as d4p
    X = x.reshape(-1, 1).astype(np.float32 if fptype == "float" else np.float64)
    if chunks is None:
        r = d4p.low_order_moments(fptype=fptype).compute(X)
    else:
        alg = d4p.low_order_moments(fptype=fptype, streaming=True)
        for c in np.array_split(X, chunks):
            alg.compute(c)
        r = alg.finalize()
    return float(r.mean[0, 0]), float(r.variance[0, 0])

rows = []
for name, x in datasets():
    with tempfile.NamedTemporaryFile(suffix=".f64", delete=False) as f:
        x.astype(np.float64).tofile(f); path = f.name
    for ftype in ("double", "float"):
        xs = x.astype(np.float32).astype(np.float64) if ftype == "float" else x
        ref_m, ref_v = exact(xs)
        def add(arm, cfg, m, v):
            rows.append(dict(data=name, type=ftype, arm=arm, cfg=cfg,
                             mean_rel=relerr(m, ref_m), var_rel=relerr(v, ref_v),
                             var=v, ref_var=float(ref_v) if ref_v is not None else None))
        if len(x) >= 2:
            for exe in ("replica", "replica_fma"):
                for B, L, order, R in itertools.product((0, 8, 128), (16, 32),
                                                        ("kernel", "rev", "shuffle"), (1, 4)):
                    if order != "kernel" and L == 32: continue
                    if B and B > len(x) // R: continue
                    m, v = replica(path, f"{HERE}/{exe}", ftype, B, L, order, 7, R)
                    add(exe, f"B={B or 'auto'} L={L} order={order} ranks={R}", m, v)
        for chunks in (None, 1, 3, 64, 1000):
            if chunks and chunks > len(x): continue
            try:
                m, v = daal(x, ftype, chunks)
                add("daal4py", f"chunks={chunks or 'batch'}", m, v)
            except Exception as e:
                rows.append(dict(data=name, type=ftype, arm="daal4py",
                                 cfg=f"chunks={chunks}", error=str(e)[:120]))
json.dump(rows, open(f"{HERE}/results.json", "w"), indent=1)

# summary: worst relative error per data/type/arm across configurations
from collections import defaultdict
agg = defaultdict(lambda: [0.0, 0.0, 0, ""])
for r in rows:
    if "error" in r: continue
    k = (r["data"], r["type"], r["arm"]); a = agg[k]
    a[2] += 1
    if r["mean_rel"] is not None: a[0] = max(a[0], r["mean_rel"])
    if r["var_rel"] is not None and r["var_rel"] > a[1]: a[1] = r["var_rel"]; a[3] = r["cfg"]
with open(f"{HERE}/summary.md", "w") as f:
    f.write("| data | type | arm | configs | max rel err mean | max rel err var | worst var config |\n|---|---|---|---|---|---|---|\n")
    for (d, t, arm), (em, ev, c, w) in sorted(agg.items()):
        f.write(f"| {d} | {t} | {arm} | {c} | {em:.2e} | {ev:.2e} | {w} |\n")
    errs = [r for r in rows if "error" in r]
    for r in errs: f.write(f"\nerror: {r['data']} {r['type']} {r['arm']} {r['cfg']}: {r['error']}\n")
print(open(f"{HERE}/summary.md").read())
