#!/usr/bin/env python3
# file: hdbscan_plot_summary.py
#===============================================================================
* Copyright contributors to the oneDAL project
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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from the benchmark run
data = [
    ("blobs_500_2d",      500,   2,  1.0000, 1.0000,   0,   0),
    ("blobs_1k_2d",      1000,   2,  1.0000, 1.0000,   0,   0),
    ("blobs_2k_2d",      2000,   2,  1.0000, 1.0000,   0,   0),
    ("blobs_3k_2d",      3000,   2,  0.9982, 0.9982,   3,   3),
    ("blobs_5k_2d",      5000,   2,  0.9994, 0.9995,   2,   1),
    ("blobs_5k_5d",      5000,   5,  1.0000, 1.0000,   0,   0),
    ("blobs_5k_10d",     5000,  10,  1.0000, 1.0000,   0,   0),
    ("blobs_5k_20d",     5000,  20,  1.0000, 1.0000,   0,   0),
    ("moons_5k_2d",      5000,   2,  0.9992, 0.9992,   1,   1),
    ("blobs_7.5k_2d",    7500,   2,  0.9994, 0.9994,   3,   3),
    ("blobs_10k_2d",    10000,   2,  0.9999, 0.9995,   1,   3),
    ("blobs_10k_5d",    10000,   5,  1.0000, 1.0000,   0,   0),
    ("blobs_10k_10d",   10000,  10,  1.0000, 1.0000,   0,   0),
    ("blobs_10k_20d",   10000,  20,  1.0000, 1.0000,   0,   0),
    ("blobs_15k_2d",    15000,   2,  0.9998, 0.9996,   2,   4),
    ("blobs_20k_2d",    20000,   2,  1.0000, 0.9999,   1,   1),
    ("blobs_20k_5d",    20000,   5,  1.0000, 1.0000,   0,   0),
    ("blobs_20k_10d",   20000,  10,  1.0000, 1.0000,   0,   0),
    ("blobs_30k_2d",    30000,   2,  0.9999, 0.9999,   2,   3),
    ("blobs_40k_2d",    40000,   2,  0.9998, 0.9999,   4,   3),
    ("blobs_50k_2d",    50000,   2,  1.0000, 0.9999,   1,   3),
    ("blobs_75k_2d",    75000,   2,  None,   0.9999,   None, 5),
    ("blobs_100k_2d",  100000,   2,  None,   1.0000,   None, 3),
]

labels = [d[0] for d in data]
ns = [d[1] for d in data]
ari_bf = [d[3] for d in data]
ari_kd = [d[4] for d in data]
err_bf = [d[5] for d in data]
err_kd = [d[6] for d in data]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 10), gridspec_kw={'hspace': 0.45})

x = np.arange(len(labels))
w = 0.35

# ---- Panel 1: ARI ----
bf_vals = [v if v is not None else 0 for v in ari_bf]
kd_vals = [v if v is not None else 0 for v in ari_kd]

bars1 = ax1.bar(x - w/2, bf_vals, w, label='oneDAL brute_force', color='#2196F3', alpha=0.85)
bars2 = ax1.bar(x + w/2, kd_vals, w, label='oneDAL kd_tree', color='#FF9800', alpha=0.85)

# Mark OOM
for j, v in enumerate(ari_bf):
    if v is None:
        bars1[j].set_color('#E0E0E0')
        bars1[j].set_edgecolor('red')
        bars1[j].set_linewidth(1.5)
        ax1.text(x[j] - w/2, 0.991, 'OOM', ha='center', va='bottom',
                fontsize=7, color='red', fontweight='bold')

ax1.axhline(y=1.0, color='green', linewidth=1.5, linestyle='--', alpha=0.6, label='Perfect (1.0)')
ax1.set_ylim(0.996, 1.001)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=50, ha='right', fontsize=9)
ax1.set_ylabel('ARI vs sklearn', fontsize=12)
ax1.set_title('Adjusted Rand Index: oneDAL vs scikit-learn  (1.0 = identical clustering)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower left', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on non-perfect bars
for j in range(len(labels)):
    for val, xoff in [(ari_bf[j], -w/2), (ari_kd[j], w/2)]:
        if val is not None and val < 1.0:
            ax1.text(x[j] + xoff, val + 0.0001, f'{val:.4f}', ha='center', va='bottom',
                    fontsize=7, rotation=90)

# ---- Panel 2: Mistake counts ----
bf_e = [v if v is not None else 0 for v in err_bf]
kd_e = [v if v is not None else 0 for v in err_kd]

bars3 = ax2.bar(x - w/2, bf_e, w, label='brute_force mistakes', color='#f44336', alpha=0.75)
bars4 = ax2.bar(x + w/2, kd_e, w, label='kd_tree mistakes', color='#9C27B0', alpha=0.75)

for j, v in enumerate(err_bf):
    if v is None:
        bars3[j].set_color('#E0E0E0')
        bars3[j].set_edgecolor('red')
        bars3[j].set_linewidth(1.5)

# Add percentage labels
for j in range(len(labels)):
    n = ns[j]
    for val, xoff, color in [(err_bf[j], -w/2, '#f44336'), (err_kd[j], w/2, '#9C27B0')]:
        if val is not None and val > 0:
            pct = val / n * 100
            ax2.text(x[j] + xoff, val + 0.15, f'{val}\n({pct:.2f}%)',
                    ha='center', va='bottom', fontsize=7, color=color)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=50, ha='right', fontsize=9)
ax2.set_ylabel('Mismatched labels', fontsize=12)
ax2.set_title('Label Disagreements vs sklearn  (0 = perfect match)  —  worst case: 5 out of 75,000 = 0.01%', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 7)

out = "/localdisk2/mkl/asolovev/oneDAL/scripts/benchmark_results/plots/00_summary.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
