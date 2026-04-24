#!/usr/bin/env python3
# file: hdbscan_plot_correctness.py
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

import subprocess
import sys
import os
import tempfile
import warnings
import gc

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

# ============================================================================
# Config
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAL_ROOT = os.path.dirname(BASE_DIR)
EXAMPLES_DIR = os.path.join(
    DAL_ROOT,
    "__release_lnx/daal/latest/examples/oneapi/dpc/_cmake_results/intel_intel64_so"
)
BF_BIN = os.path.join(EXAMPLES_DIR, "hdbscan_brute_force_batch")
KD_BIN = os.path.join(EXAMPLES_DIR, "hdbscan_kd_tree_batch")
OUT_DIR = os.path.join(BASE_DIR, "benchmark_results", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# All configs from the benchmark
CONFIGS = [
    # (name, N, D, min_cluster_size, min_samples)
    ("blobs", 500, 2, 10, 5),
    ("blobs", 1000, 2, 10, 5),
    ("blobs", 2000, 2, 15, 10),
    ("blobs", 3000, 2, 15, 10),
    ("blobs", 5000, 2, 15, 10),
    ("blobs", 5000, 5, 15, 10),
    ("blobs", 5000, 10, 15, 10),
    ("blobs", 5000, 20, 15, 10),
    ("moons", 5000, 2, 15, 10),
    ("blobs", 7500, 2, 15, 10),
    ("blobs", 10000, 2, 15, 10),
    ("blobs", 10000, 5, 15, 10),
    ("blobs", 10000, 10, 15, 10),
    ("blobs", 10000, 20, 15, 10),
    ("blobs", 15000, 2, 20, 10),
    ("blobs", 20000, 2, 20, 10),
    ("blobs", 20000, 5, 20, 10),
    ("blobs", 20000, 10, 20, 10),
    ("blobs", 30000, 2, 25, 15),
    ("blobs", 40000, 2, 25, 15),
    ("blobs", 50000, 2, 25, 15),
    ("blobs", 75000, 2, 30, 20),
    ("blobs", 100000, 2, 30, 20),
]

# Distinct colors for clusters (up to 15 clusters + noise)
CLUSTER_CMAP = plt.cm.tab20
NOISE_COLOR = (0.7, 0.7, 0.7, 0.4)  # light gray, semi-transparent


def generate_dataset(name, n, d):
    rng = np.random.RandomState(42)
    if name == "blobs":
        n_centers = max(3, min(n // 300, 10))
        X, _ = make_blobs(n_samples=n, n_features=d, centers=n_centers,
                          cluster_std=1.0, random_state=42)
    elif name == "moons":
        X, _ = make_moons(n_samples=n, noise=0.08, random_state=42)
        if d > 2:
            extra = rng.randn(n, d - 2) * 0.5
            X = np.hstack([X, extra])
    return X.astype(np.float32)


def save_csv(data, path):
    with open(path, "w") as f:
        for row in data:
            f.write(",".join(f"{v:.8f}" for v in row) + "\n")


def run_onedal(binary, data_path, mcs, ms, device="opencl:cpu", timeout=600):
    env = os.environ.copy()
    env["HDBSCAN_DATA_PATH"] = data_path
    env["HDBSCAN_MIN_CLUSTER_SIZE"] = str(mcs)
    env["HDBSCAN_MIN_SAMPLES"] = str(ms)
    env["ONEAPI_DEVICE_SELECTOR"] = device
    try:
        result = subprocess.run(
            [binary], capture_output=True, text=True, timeout=timeout, env=env
        )
        output = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, Exception):
        return None, None, None

    time_ms = n_clusters = labels = None
    for line in output.strip().split("\n"):
        line = line.strip()
        if line.startswith("Time:"):
            time_ms = float(line.split(":")[1].strip().replace(" ms", ""))
        elif line.startswith("Cluster count:"):
            n_clusters = int(line.split(":")[1].strip())
        elif line.startswith("Labels:"):
            labels = np.array([int(x) for x in line.replace("Labels:", "").strip().split()])
    return time_ms, n_clusters, labels


def run_sklearn(data, mcs, ms, algorithm="brute"):
    import time
    hdb = SklearnHDBSCAN(algorithm=algorithm, min_cluster_size=mcs,
                          min_samples=ms, copy=True)
    t0 = time.time()
    labels = hdb.fit_predict(data)
    t1 = time.time()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return (t1 - t0) * 1000, n_clusters, labels


def label_colors(labels):
    """Map cluster labels to colors. -1 = noise (gray)."""
    colors = np.zeros((len(labels), 4))
    unique = sorted(set(labels))
    cluster_ids = [u for u in unique if u >= 0]
    for i, lab in enumerate(labels):
        if lab == -1:
            colors[i] = NOISE_COLOR
        else:
            idx = cluster_ids.index(lab) if lab in cluster_ids else 0
            colors[i] = CLUSTER_CMAP(idx / max(len(cluster_ids), 1))
    return colors


def remap_labels(ref_labels, target_labels):
    """Remap target_labels so that cluster IDs match ref_labels as closely as possible.
    Uses the Hungarian-like greedy matching on label overlap."""
    ref_unique = sorted(set(ref_labels[ref_labels >= 0]))
    tgt_unique = sorted(set(target_labels[target_labels >= 0]))

    if not ref_unique or not tgt_unique:
        return target_labels.copy()

    # Build overlap matrix
    mapping = {}
    used_ref = set()
    for t in tgt_unique:
        t_mask = target_labels == t
        best_ref = -1
        best_overlap = 0
        for r in ref_unique:
            if r in used_ref:
                continue
            overlap = np.sum((ref_labels == r) & t_mask)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ref = r
        if best_ref >= 0:
            mapping[t] = best_ref
            used_ref.add(best_ref)
        else:
            mapping[t] = t + 100  # unmapped cluster

    remapped = target_labels.copy()
    for old, new in mapping.items():
        remapped[target_labels == old] = new
    return remapped


def plot_dataset(X, sk_labels, dal_bf_labels, dal_kd_labels,
                 sk_time, bf_time, kd_time,
                 dataset_label, n, d, mcs, ms, out_path):
    """Create a 4-panel figure for one dataset."""

    # Use first 2 dims for plotting; if D > 2, note it
    x_plot = X[:, 0]
    y_plot = X[:, 1]
    high_d = d > 2

    # Remap oneDAL labels to match sklearn label numbering
    if dal_bf_labels is not None:
        dal_bf_remapped = remap_labels(sk_labels, dal_bf_labels)
    else:
        dal_bf_remapped = None
    if dal_kd_labels is not None:
        dal_kd_remapped = remap_labels(sk_labels, dal_kd_labels)
    else:
        dal_kd_remapped = None

    # Compute agreement/mistakes
    def compute_diff(ref, tgt):
        if tgt is None:
            return None, None, 0, 0
        agree = ref == tgt
        disagree = ~agree
        n_agree = np.sum(agree)
        n_disagree = np.sum(disagree)
        return agree, disagree, n_agree, n_disagree

    bf_agree, bf_disagree, bf_n_agree, bf_n_disagree = compute_diff(sk_labels, dal_bf_remapped)
    kd_agree, kd_disagree, kd_n_agree, kd_n_disagree = compute_diff(sk_labels, dal_kd_remapped)

    # ARI scores
    ari_bf = adjusted_rand_score(sk_labels, dal_bf_labels) if dal_bf_labels is not None else None
    ari_kd = adjusted_rand_score(sk_labels, dal_kd_labels) if dal_kd_labels is not None else None

    n_clusters_sk = len(set(sk_labels)) - (1 if -1 in sk_labels else 0)
    n_noise_sk = np.sum(sk_labels == -1)

    # Point size based on N
    if n <= 1000:
        s = 12
    elif n <= 5000:
        s = 5
    elif n <= 20000:
        s = 2
    else:
        s = 0.5

    # Figure: 2 rows x 2 cols
    # Row 1: sklearn reference | oneDAL brute_force labels
    # Row 2: oneDAL kd_tree labels | Mistakes overlay
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(
        f"{dataset_label}  —  N={n:,}  D={d}  min_cluster_size={mcs}  min_samples={ms}\n"
        f"sklearn: {n_clusters_sk} clusters, {n_noise_sk} noise pts"
        + (f"  |  Projected to first 2 of {d} dimensions" if high_d else ""),
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.15, top=0.92, bottom=0.05,
                           left=0.05, right=0.95)

    # ---- Panel 1: sklearn reference ----
    ax1 = fig.add_subplot(gs[0, 0])
    colors_sk = label_colors(sk_labels)
    ax1.scatter(x_plot, y_plot, c=colors_sk, s=s, alpha=0.8, edgecolors='none')
    ax1.set_title(f"sklearn (reference)\n{n_clusters_sk} clusters  |  {sk_time:.0f} ms", fontsize=12)
    ax1.set_xlabel("Feature 0")
    ax1.set_ylabel("Feature 1")

    # ---- Panel 2: oneDAL brute_force ----
    ax2 = fig.add_subplot(gs[0, 1])
    if dal_bf_labels is not None:
        colors_bf = label_colors(dal_bf_remapped)
        ax2.scatter(x_plot, y_plot, c=colors_bf, s=s, alpha=0.8, edgecolors='none')
        n_cl_bf = len(set(dal_bf_labels)) - (1 if -1 in dal_bf_labels else 0)
        ax2.set_title(
            f"oneDAL brute_force\n{n_cl_bf} clusters  |  {bf_time:.0f} ms  |  "
            f"ARI={ari_bf:.4f}  |  {bf_n_disagree} mistakes ({bf_n_disagree/n*100:.2f}%)",
            fontsize=12,
            color='green' if bf_n_disagree == 0 else ('orange' if bf_n_disagree / n < 0.01 else 'red')
        )
    else:
        ax2.text(0.5, 0.5, "OOM / Not Run\n(N² memory exceeded)", transform=ax2.transAxes,
                 ha='center', va='center', fontsize=16, color='red')
        ax2.set_title("oneDAL brute_force — N/A", fontsize=12)
    ax2.set_xlabel("Feature 0")
    ax2.set_ylabel("Feature 1")

    # ---- Panel 3: oneDAL kd_tree ----
    ax3 = fig.add_subplot(gs[1, 0])
    if dal_kd_labels is not None:
        colors_kd = label_colors(dal_kd_remapped)
        ax3.scatter(x_plot, y_plot, c=colors_kd, s=s, alpha=0.8, edgecolors='none')
        n_cl_kd = len(set(dal_kd_labels)) - (1 if -1 in dal_kd_labels else 0)
        ax3.set_title(
            f"oneDAL kd_tree\n{n_cl_kd} clusters  |  {kd_time:.0f} ms  |  "
            f"ARI={ari_kd:.4f}  |  {kd_n_disagree} mistakes ({kd_n_disagree/n*100:.2f}%)",
            fontsize=12,
            color='green' if kd_n_disagree == 0 else ('orange' if kd_n_disagree / n < 0.01 else 'red')
        )
    else:
        ax3.text(0.5, 0.5, "OOM / Not Run", transform=ax3.transAxes,
                 ha='center', va='center', fontsize=16, color='red')
        ax3.set_title("oneDAL kd_tree — N/A", fontsize=12)
    ax3.set_xlabel("Feature 0")
    ax3.set_ylabel("Feature 1")

    # ---- Panel 4: Mistakes overlay ----
    ax4 = fig.add_subplot(gs[1, 1])

    # Background: all points in light gray
    ax4.scatter(x_plot, y_plot, c='lightgray', s=s * 0.5, alpha=0.3, edgecolors='none')

    legend_handles = []

    # Overlay BF mistakes in red
    if bf_disagree is not None and np.any(bf_disagree):
        ax4.scatter(x_plot[bf_disagree], y_plot[bf_disagree],
                    c='red', s=s * 4, alpha=0.7, edgecolors='none', marker='x', linewidths=0.5)
        legend_handles.append(Line2D([0], [0], marker='x', color='red', linestyle='None',
                                     markersize=6, label=f'BF mistakes ({bf_n_disagree})'))

    # Overlay KD mistakes in blue
    if kd_disagree is not None and np.any(kd_disagree):
        ax4.scatter(x_plot[kd_disagree], y_plot[kd_disagree],
                    c='blue', s=s * 4, alpha=0.7, edgecolors='none', marker='+', linewidths=0.5)
        legend_handles.append(Line2D([0], [0], marker='+', color='blue', linestyle='None',
                                     markersize=6, label=f'KD mistakes ({kd_n_disagree})'))

    if not legend_handles:
        ax4.text(0.5, 0.5, "PERFECT MATCH\nNo mistakes",
                 transform=ax4.transAxes, ha='center', va='center',
                 fontsize=20, color='green', fontweight='bold')

    # Summary text
    summary_lines = []
    if ari_bf is not None:
        summary_lines.append(f"BF: ARI={ari_bf:.4f}, {bf_n_disagree}/{n} wrong ({bf_n_disagree/n*100:.2f}%)")
    else:
        summary_lines.append("BF: N/A (OOM)")
    if ari_kd is not None:
        summary_lines.append(f"KD: ARI={ari_kd:.4f}, {kd_n_disagree}/{n} wrong ({kd_n_disagree/n*100:.2f}%)")
    else:
        summary_lines.append("KD: N/A")

    ax4.set_title(
        f"Mistakes vs sklearn\n" + "  |  ".join(summary_lines),
        fontsize=11
    )
    if legend_handles:
        ax4.legend(handles=legend_handles, loc='upper right', fontsize=10)
    ax4.set_xlabel("Feature 0")
    ax4.set_ylabel("Feature 1")

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ari_bf, ari_kd, bf_n_disagree, kd_n_disagree


def main():
    if not os.path.exists(BF_BIN) or not os.path.exists(KD_BIN):
        print(f"ERROR: Binaries not found at {EXAMPLES_DIR}")
        sys.exit(1)

    print(f"Output directory: {OUT_DIR}")
    print(f"Configs: {len(CONFIGS)} datasets")
    print()

    summary_rows = []

    for i, (ds_name, n, d, mcs, ms) in enumerate(CONFIGS):
        label = f"{ds_name}_{n}_{d}d"
        print(f"[{i+1}/{len(CONFIGS)}] {label} (N={n:,}, D={d}) ...", end=" ", flush=True)

        data = generate_dataset(ds_name, n, d)

        tmpfile = os.path.join(tempfile.gettempdir(), f"hdbscan_plot_{n}_{d}.csv")
        save_csv(data, tmpfile)

        # sklearn reference (use brute for N <= 50k, kd_tree for larger)
        if n <= 50000:
            sk_time, sk_nc, sk_labels = run_sklearn(data, mcs, ms, "brute")
        else:
            sk_time, sk_nc, sk_labels = run_sklearn(data, mcs, ms, "kd_tree")

        # oneDAL brute_force (skip if N > 50k — OOM)
        bf_time, bf_nc, bf_labels = (None, None, None)
        if n <= 50000:
            bf_time, bf_nc, bf_labels = run_onedal(BF_BIN, tmpfile, mcs, ms)

        # oneDAL kd_tree
        kd_time, kd_nc, kd_labels = run_onedal(KD_BIN, tmpfile, mcs, ms)

        out_path = os.path.join(OUT_DIR, f"{i+1:02d}_{label}.png")
        ari_bf, ari_kd, bf_mistakes, kd_mistakes = plot_dataset(
            data, sk_labels, bf_labels, kd_labels,
            sk_time or 0, bf_time or 0, kd_time or 0,
            label, n, d, mcs, ms, out_path
        )

        status = "OK"
        if bf_labels is not None and bf_mistakes > 0:
            status = f"BF:{bf_mistakes}"
        if kd_labels is not None and kd_mistakes > 0:
            status += f" KD:{kd_mistakes}"

        summary_rows.append((label, n, d, ari_bf, ari_kd,
                             bf_mistakes if bf_labels is not None else None,
                             kd_mistakes if kd_labels is not None else None,
                             out_path))

        print(f"ARI(BF)={'N/A' if ari_bf is None else f'{ari_bf:.4f}'}, "
              f"ARI(KD)={'N/A' if ari_kd is None else f'{ari_kd:.4f}'}, "
              f"saved {os.path.basename(out_path)}")

        try:
            os.unlink(tmpfile)
        except OSError:
            pass
        del data
        gc.collect()

    # ========================================================================
    # Summary image: all ARIs + mistake counts
    # ========================================================================
    print()
    print("Generating summary plot...")

    labels_list = [r[0] for r in summary_rows]
    ns = [r[1] for r in summary_rows]
    ds = [r[2] for r in summary_rows]
    ari_bf_list = [r[3] for r in summary_rows]
    ari_kd_list = [r[4] for r in summary_rows]
    bf_m_list = [r[5] for r in summary_rows]
    kd_m_list = [r[6] for r in summary_rows]

    fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'hspace': 0.35})

    # Panel 1: ARI scores
    ax = axes[0]
    x_pos = np.arange(len(labels_list))
    bar_width = 0.35

    ari_bf_vals = [v if v is not None else 0 for v in ari_bf_list]
    ari_kd_vals = [v if v is not None else 0 for v in ari_kd_list]
    bf_available = [v is not None for v in ari_bf_list]
    kd_available = [v is not None for v in ari_kd_list]

    bars1 = ax.bar(x_pos - bar_width/2, ari_bf_vals, bar_width,
                   label='oneDAL brute_force', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x_pos + bar_width/2, ari_kd_vals, bar_width,
                   label='oneDAL kd_tree', color='#FF9800', alpha=0.8)

    # Mark unavailable (OOM) bars
    for j, avail in enumerate(bf_available):
        if not avail:
            bars1[j].set_color('lightgray')
            bars1[j].set_edgecolor('red')
            bars1[j].set_linewidth(1.5)
            ax.text(x_pos[j] - bar_width/2, 0.02, 'OOM', ha='center', va='bottom',
                    fontsize=7, color='red', rotation=90)

    ax.set_ylim(0.99, 1.001)
    ax.axhline(y=1.0, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Perfect (1.0)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('ARI vs sklearn', fontsize=12)
    ax.set_title('Adjusted Rand Index: oneDAL vs scikit-learn (closer to 1.0 = better)', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Mistake counts (log scale)
    ax2 = axes[1]
    bf_m_vals = [v if v is not None else 0 for v in bf_m_list]
    kd_m_vals = [v if v is not None else 0 for v in kd_m_list]

    bars3 = ax2.bar(x_pos - bar_width/2, bf_m_vals, bar_width,
                    label='oneDAL brute_force mistakes', color='#f44336', alpha=0.7)
    bars4 = ax2.bar(x_pos + bar_width/2, kd_m_vals, bar_width,
                    label='oneDAL kd_tree mistakes', color='#9C27B0', alpha=0.7)

    # Add percentage labels on bars
    for j in range(len(labels_list)):
        n_pts = ns[j]
        if bf_m_list[j] is not None and bf_m_list[j] > 0:
            pct = bf_m_list[j] / n_pts * 100
            ax2.text(x_pos[j] - bar_width/2, bf_m_vals[j], f'{pct:.1f}%',
                     ha='center', va='bottom', fontsize=7, rotation=90)
        if kd_m_list[j] is not None and kd_m_list[j] > 0:
            pct = kd_m_list[j] / n_pts * 100
            ax2.text(x_pos[j] + bar_width/2, kd_m_vals[j], f'{pct:.1f}%',
                     ha='center', va='bottom', fontsize=7, rotation=90)

    ax2.set_yscale('symlog', linthresh=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Number of mismatched labels', fontsize=12)
    ax2.set_title('Label Disagreements vs sklearn (0 = perfect match)', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    summary_path = os.path.join(OUT_DIR, "00_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary saved: {summary_path}")

    # Print final table
    print()
    print("=" * 100)
    print(f"{'Dataset':<20} {'N':>7} {'D':>3} | "
          f"{'ARI BF':>8} {'ARI KD':>8} | "
          f"{'BF err':>7} {'BF %':>7} | "
          f"{'KD err':>7} {'KD %':>7}")
    print("-" * 100)
    for row in summary_rows:
        label, n, d, ari_bf, ari_kd, bf_m, kd_m, _ = row
        print(f"{label:<20} {n:>7,} {d:>3} | "
              f"{'  N/A   ' if ari_bf is None else f'{ari_bf:>8.4f}'} "
              f"{'  N/A   ' if ari_kd is None else f'{ari_kd:>8.4f}'} | "
              f"{'  N/A  ' if bf_m is None else f'{bf_m:>7,}'} "
              f"{'  N/A  ' if bf_m is None else f'{bf_m/n*100:>6.2f}%'} | "
              f"{'  N/A  ' if kd_m is None else f'{kd_m:>7,}'} "
              f"{'  N/A  ' if kd_m is None else f'{kd_m/n*100:>6.2f}%'}")

    print()
    print(f"All plots saved to: {OUT_DIR}")
    print(f"Total images: {len(CONFIGS) + 1} (1 summary + {len(CONFIGS)} per-dataset)")


if __name__ == "__main__":
    main()
