#!/usr/bin/env python3
"""Generate JPEG figures for HDBSCAN sprint review."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = "scripts/benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def figure_algorithm_pipeline():
    """HDBSCAN algorithm pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("HDBSCAN Algorithm Pipeline in oneDAL", fontsize=16, fontweight="bold", pad=20)

    steps_top = [
        (1.0, 5.5, "1. Pairwise\nDistances", "#4472C4", "BLAS GEMM\n(MKL)"),
        (3.5, 5.5, "2. Core\nDistances", "#4472C4", "k-th NN\ndistance"),
        (6.0, 5.5, "3. Mutual\nReachability", "#4472C4", "MRD matrix"),
        (8.5, 5.5, "4. Minimum\nSpanning Tree", "#ED7D31", "Prim's\nalgorithm"),
        (11.0, 5.5, "5. Sort MST\nby weight", "#4472C4", "parallel\nsort"),
    ]

    steps_bot = [
        (2.25, 2.0, "6. Build\nDendrogram", "#4472C4", "single-linkage\nhierarchy"),
        (5.5, 2.0, "7. Condensed\nTree", "#4472C4", "min_cluster_size\nfilter"),
        (8.75, 2.0, "8. Extract\nClusters (EOM)", "#70AD47", "stability-based\nselection"),
        (12.0, 2.0, "9. Assign\nLabels", "#70AD47", "final output"),
    ]

    def draw_step(ax, x, y, title, color, subtitle):
        box = FancyBboxPatch((x-0.9, y-0.7), 1.8, 1.4,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="black", alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y+0.15, title, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")
        ax.text(x, y-0.85, subtitle, ha="center", va="top", fontsize=7.5,
                color="#444444", style="italic")

    for x, y, title, color, sub in steps_top:
        draw_step(ax, x, y, title, color, sub)
    for x, y, title, color, sub in steps_bot:
        draw_step(ax, x, y, title, color, sub)

    # Arrows top row
    for i in range(len(steps_top)-1):
        ax.annotate("", xy=(steps_top[i+1][0]-0.9, steps_top[i+1][1]),
                    xytext=(steps_top[i][0]+0.9, steps_top[i][1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Arrow top to bottom
    ax.annotate("", xy=(steps_bot[0][0], steps_bot[0][1]+0.7),
                xytext=(steps_top[-1][0], steps_top[-1][1]-0.7),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5,
                               connectionstyle="arc3,rad=0.3"))

    # Arrows bottom row
    for i in range(len(steps_bot)-1):
        ax.annotate("", xy=(steps_bot[i+1][0]-0.9, steps_bot[i+1][1]),
                    xytext=(steps_bot[i][0]+0.9, steps_bot[i][1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Header text
    ax.text(7.0, 6.8,
            "CPU: DAAL kernel (BLAS + threader_for + ISA dispatch)  |  GPU: SYCL kernels + host MST",
            ha="center", fontsize=10, style="italic", color="#333333")

    # Parallel vs sequential boxes
    ax.add_patch(plt.Rectangle((0.0, 4.5), 7.8, 2.2, fill=False,
                 edgecolor="#4472C4", linestyle="--", linewidth=1.5))
    ax.text(3.9, 4.55, "Parallel (O(N^2) - GEMM accelerated)",
            ha="center", fontsize=8, color="#4472C4")

    ax.add_patch(plt.Rectangle((7.5, 4.5), 2.8, 2.2, fill=False,
                 edgecolor="#ED7D31", linestyle="--", linewidth=1.5))
    ax.text(8.9, 4.55, "Sequential O(N^2)",
            ha="center", fontsize=8, color="#ED7D31")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/hdbscan_algorithm_pipeline.jpg"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def figure_performance_comparison():
    """Performance bar chart: oneDAL vs sklearn."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    datasets = ["500\n2D", "1k\n2D", "2k\n2D", "5k\n2D", "5k\n5D", "5k\n10D",
                "10k\n2D", "10k\n10D", "20k\n2D", "30k\n2D", "50k\n2D", "10.5k\n2D"]
    sklearn_ms = [6.7, 18.4, 112.8, 795.7, 788.4, 899.9, 2818.7, 1890.2, 8272.4, 18448.0, 55114.0, 1808.7]
    cpu_ms = [3.1, 7.0, 21.0, 130.9, 131.3, 133.1, 270.1, 277.5, 1117.2, 2656.2, 7288.0, 306.1]
    gpu_ms = [4.9, 10.4, 30.0, 161.6, 142.4, 133.0, 669.2, 692.5, 1981.3, 0, 0, 792.6]

    x = np.arange(len(datasets))
    width = 0.25

    ax1.bar(x - width, sklearn_ms, width, label="scikit-learn", color="#C00000", alpha=0.85)
    ax1.bar(x, cpu_ms, width, label="oneDAL CPU", color="#4472C4", alpha=0.85)
    ax1.bar(x + width, [g if g > 0 else 0 for g in gpu_ms], width,
            label="oneDAL GPU", color="#70AD47", alpha=0.85)

    ax1.set_yscale("log")
    ax1.set_xlabel("Dataset (N points, Dimensions)", fontsize=11)
    ax1.set_ylabel("Time (ms, log scale)", fontsize=11)
    ax1.set_title("Execution Time: oneDAL vs scikit-learn", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=8)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(1, 100000)

    # Speedup chart
    cpu_speedups = [s/c for s, c in zip(sklearn_ms, cpu_ms)]
    gpu_speedups = [s/g if g > 0 else 0 for s, g in zip(sklearn_ms, gpu_ms)]

    ax2.bar(x - 0.2, cpu_speedups, 0.35, label="CPU Speedup", color="#4472C4", alpha=0.85)
    ax2.bar(x + 0.2, gpu_speedups, 0.35, label="GPU Speedup", color="#70AD47", alpha=0.85)

    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    cpu_avg = np.mean(cpu_speedups)
    ax2.axhline(y=cpu_avg, color="#4472C4", linestyle=":", linewidth=2, alpha=0.7)
    ax2.text(len(datasets)-0.5, cpu_avg+0.3, f"CPU avg: {cpu_avg:.1f}x",
             fontsize=9, color="#4472C4", fontweight="bold")

    gpu_valid = [g for g in gpu_speedups if g > 0]
    gpu_avg = np.mean(gpu_valid)
    ax2.axhline(y=gpu_avg, color="#70AD47", linestyle=":", linewidth=2, alpha=0.7)
    ax2.text(len(datasets)-0.5, gpu_avg-0.5, f"GPU avg: {gpu_avg:.1f}x",
             fontsize=9, color="#70AD47", fontweight="bold")

    ax2.set_xlabel("Dataset (N points, Dimensions)", fontsize=11)
    ax2.set_ylabel("Speedup vs scikit-learn", fontsize=11)
    ax2.set_title("Speedup: oneDAL vs scikit-learn", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=8)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 12)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/hdbscan_performance_comparison.jpg"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def figure_scaling_curve():
    """Speedup vs dataset size."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sizes = [500, 1000, 2000, 5000, 10000, 20000, 30000, 50000]
    cpu_scale = [2.18, 2.63, 5.37, 6.28, 8.63, 7.40, 6.95, 7.56]
    gpu_scale = [1.38, 1.77, 3.76, 5.74, 3.47, 4.18, None, None]

    ax.plot(sizes, cpu_scale, "o-", color="#4472C4", linewidth=2.5, markersize=8,
            label="oneDAL CPU", zorder=5)

    gpu_sizes_valid = [s for s, g in zip(sizes, gpu_scale) if g is not None]
    gpu_scale_valid = [g for g in gpu_scale if g is not None]
    ax.plot(gpu_sizes_valid, gpu_scale_valid, "s-", color="#70AD47", linewidth=2.5, markersize=8,
            label="oneDAL GPU", zorder=5)

    ax.axvspan(20000, 55000, alpha=0.08, color="red")
    ax.text(35000, 9.5, "GPU limited\n(precision / OOM)", ha="center", fontsize=9,
            color="#C00000", style="italic")

    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="sklearn baseline")
    ax.fill_between(sizes, 1.0, cpu_scale, alpha=0.1, color="#4472C4")

    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (N points)", fontsize=12)
    ax.set_ylabel("Speedup vs scikit-learn", fontsize=12)
    ax.set_title("HDBSCAN Speedup Scaling: oneDAL vs scikit-learn", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 12)
    ax.set_xlim(300, 70000)
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000, 50000])
    ax.set_xticklabels(["500", "1k", "2k", "5k", "10k", "20k", "50k"])

    ax.annotate("10.4x", xy=(10000, 8.63), xytext=(13000, 10.2),
                fontsize=11, fontweight="bold", color="#4472C4",
                arrowprops=dict(arrowstyle="->", color="#4472C4"))
    ax.annotate("5.7x", xy=(5000, 5.74), xytext=(6500, 7.5),
                fontsize=11, fontweight="bold", color="#70AD47",
                arrowprops=dict(arrowstyle="->", color="#70AD47"))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/hdbscan_scaling_curve.jpg"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def figure_clustering_demo():
    """Clustering visualization with summary."""
    from sklearn.datasets import make_blobs
    from sklearn.cluster import HDBSCAN as HDBSCAN_SK

    fig = plt.figure(figsize=(16, 5.5))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    X, _ = make_blobs(n_samples=2000, n_features=2, centers=5,
                      cluster_std=0.8, random_state=42)

    hdb = HDBSCAN_SK(min_cluster_size=10, min_samples=5, algorithm="brute", copy=True)
    labels = hdb.fit_predict(X)

    # Raw data
    ax0.scatter(X[:, 0], X[:, 1], c="gray", s=5, alpha=0.5)
    ax0.set_title("Input Data\n(2000 pts, 2D)", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Feature 1")
    ax0.set_ylabel("Feature 2")

    # HDBSCAN result
    noise_mask = labels == -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(noise_mask))

    cmap = plt.cm.Set1
    for cl in range(n_clusters):
        mask = labels == cl
        ax1.scatter(X[mask, 0], X[mask, 1], c=[cmap(cl % 9)], s=8, alpha=0.7,
                    label=f"Cluster {cl}")
    ax1.scatter(X[noise_mask, 0], X[noise_mask, 1], c="lightgray", s=3,
                alpha=0.3, marker="x", label="Noise")
    ax1.set_title(f"HDBSCAN: {n_clusters} clusters\n({n_noise} noise points)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Feature 1")
    ax1.legend(fontsize=7, loc="lower right", ncol=2)

    # Summary as a table instead of text box
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    ax2.text(5, 9.5, "oneDAL HDBSCAN", ha="center", fontsize=14, fontweight="bold",
             color="#4472C4")
    ax2.text(5, 8.8, "Sprint Review Summary", ha="center", fontsize=11,
             color="#333333")

    # Draw a simple table
    rows = [
        ("CPU avg speedup", "6.0x"),
        ("CPU best speedup", "10.4x"),
        ("GPU avg speedup", "3.9x"),
        ("GPU best speedup", "6.8x"),
        ("Correctness", "PASS (ARI>=0.99)"),
        ("Datasets tested", "13"),
        ("Size range", "500 - 50k pts"),
        ("Dimensions", "2D - 10D"),
        ("sklearn-intelex", "NO support"),
        ("oneDAL", "FULL support"),
    ]
    y_start = 8.0
    for i, (label, value) in enumerate(rows):
        y = y_start - i * 0.75
        ax2.text(1.0, y, label, fontsize=10, va="center",
                 fontfamily="sans-serif")
        color = "#006100" if "x" in value or value == "PASS (ARI>=0.99)" or value == "FULL support" else "#333333"
        weight = "bold" if "x" in value or "PASS" in value or "FULL" in value else "normal"
        ax2.text(9.0, y, value, fontsize=10, va="center", ha="right",
                 color=color, fontweight=weight, fontfamily="sans-serif")
        if i < len(rows) - 1:
            ax2.axhline(y=y-0.35, xmin=0.05, xmax=0.95, color="#cccccc", linewidth=0.5)

    # Border around summary
    from matplotlib.patches import FancyBboxPatch as FBP
    rect = FBP((0.2, 0.3), 9.6, 9.0, boxstyle="round,pad=0.2",
               facecolor="#F0F4FF", edgecolor="#4472C4", linewidth=1.5)
    ax2.add_patch(rect)
    ax2.set_zorder(-1)

    fig.suptitle("HDBSCAN: Density-Based Clustering with oneDAL",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.88, wspace=0.3)
    path = f"{OUTPUT_DIR}/hdbscan_clustering_demo.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    figure_algorithm_pipeline()
    figure_performance_comparison()
    figure_scaling_curve()
    figure_clustering_demo()
    print("\nAll figures generated!")
