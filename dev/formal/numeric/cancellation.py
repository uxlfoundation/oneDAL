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

"""Emulates in IEEE double the DAAL CPU online merge (expanded term) and the factored
term, chunked like daal4py streaming, against an exact rational reference."""
from fractions import Fraction
import numpy as np
rng = np.random.default_rng(20261111)
def welford_block(c):
    n=len(c); m=np.float64(c.sum())/n; return n, c.sum(), float(((c-m)**2).sum())
def merge(x, chunks, expanded):
    n1=0; s1=0.0; ssc=0.0
    for c in np.array_split(x, chunks):
        n2, s2, q2 = welford_block(c)
        if n1 == 0: n1, s1, ssc = n2, s2, q2; continue
        coeff = float(n1*n2)/float(n1+n2); m1 = s1/n1; m2 = s2/n2
        t = (m1*m1 + m2*m2 - 2*m1*m2) if expanded else (m2-m1)*(m2-m1)
        ssc += q2 + coeff*t; s1 += s2; n1 += n2
    return ssc/(n1-1)
print("| data | chunks | rel err expanded (as in DAAL) | rel err factored |")
print("|---|---|---|---|")
for name, x in [("offset1e8_sd1e-3_n20000", 1e8 + rng.normal(0, 1e-3, 20000)),
                ("offset1e4_sd1e-2_n20000", 1e4 + rng.normal(0, 1e-2, 20000)),
                ("normal_0_1_n20000", rng.normal(0, 1, 20000))]:
    q=[Fraction(float(v)) for v in x]; m=sum(q)/len(q); ref=sum((v-m)**2 for v in q)/(len(q)-1)
    for ch in (3, 64, 1000):
        e=[float(abs(Fraction(merge(x, ch, ex)) - ref)/ref) for ex in (True, False)]
        print(f"| {name} | {ch} | {e[0]:.2e} | {e[1]:.2e} |")
