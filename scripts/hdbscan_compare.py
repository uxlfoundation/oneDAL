"""
Generate a synthetic dataset, run sklearn HDBSCAN, save results for comparison with oneDAL.

Usage:
    python scripts/hdbscan_compare.py [--rows 10000] [--cols 2] [--min-cluster-size 15] [--min-samples 10]
    python scripts/hdbscan_compare.py --check

Outputs (in data/):
    hdbscan_bench.csv          - input data
    hdbscan_bench_labels.csv   - sklearn cluster labels (-1 = noise)
    hdbscan_bench_summary.txt  - summary stats
    hdbscan_bench_data.png     - scatter plot of generated data
    hdbscan_bench_compare.png  - side-by-side comparison of sklearn vs oneDAL
"""

import argparse
import numpy as np


def plot_labels(ax, X, labels, title):
    """Plot 2D scatter with cluster coloring."""
    unique_labels = sorted(set(labels))
    import matplotlib.cm as cm
    n_clusters = len([l for l in unique_labels if l >= 0])
    colors = cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    for label in unique_labels:
        mask = labels == label
        if label == -1:
            ax.scatter(X[mask, 0], X[mask, 1], c='lightgray', s=5, alpha=0.5, label='noise')
        else:
            color_idx = label % len(colors)
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[color_idx]], s=8, alpha=0.7,
                       label=f'cluster {label}')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    from sklearn.datasets import make_blobs, make_moons, make_circles
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    import time

    np.random.seed(args.seed)

    # Generate interesting multi-shape dataset for 2D visualization
    n_per_shape = args.rows // 5

    # Component 1: Two moons
    X_moons, y_moons = make_moons(n_samples=n_per_shape, noise=0.06, random_state=args.seed)
    X_moons = X_moons * 3 + np.array([0, 8])
    y_moons[:] = 0
    y_moons[X_moons.shape[0] // 2:] = 1

    # Component 2: Concentric circles
    X_circles, y_circles = make_circles(n_samples=n_per_shape, noise=0.05, factor=0.5,
                                         random_state=args.seed)
    X_circles = X_circles * 3 + np.array([8, 8])
    y_circles[:] = 2
    y_circles[X_circles.shape[0] // 2:] = 3

    # Component 3: Dense blobs
    X_blobs, y_blobs = make_blobs(
        n_samples=n_per_shape * 2,
        n_features=2,
        centers=np.array([[0, 0], [4, 0], [8, 0], [12, 0]]),
        cluster_std=0.5,
        random_state=args.seed,
    )
    y_blobs += 4  # labels 4,5,6,7

    # Component 4: Elongated Gaussian
    X_elong = np.random.randn(n_per_shape, 2) @ np.array([[2.5, 0.8], [0.8, 0.3]]) + np.array(
        [14, 8]
    )
    y_elong = np.full(n_per_shape, 8)

    # Combine
    X = np.vstack([X_moons, X_circles, X_blobs, X_elong])
    y_true = np.concatenate([y_moons, y_circles, y_blobs, y_elong])

    # If cols > 2, pad with random features (but keep first 2 for plotting)
    if args.cols > 2:
        X_extra = np.random.randn(X.shape[0], args.cols - 2) * 0.5
        X = np.hstack([X, X_extra])

    # Add ~5% uniform noise points
    n_noise = int(X.shape[0] * 0.05)
    noise = np.random.uniform(
        low=X.min(axis=0) - 3,
        high=X.max(axis=0) + 3,
        size=(n_noise, X.shape[1]),
    )
    X = np.vstack([X, noise])
    y_true = np.concatenate([y_true, np.full(n_noise, -1)])

    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y_true = y_true[perm]

    print(f"Dataset: {X.shape[0]} rows x {X.shape[1]} cols")
    print(f"  Shapes: 2 moons, 2 circles, 4 blobs, 1 elongated + {n_noise} noise points")

    # Run sklearn HDBSCAN with timing
    hdb = HDBSCAN(
        algorithm='brute',
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    # Warmup
    _ = hdb.fit_predict(X[:100])

    t0 = time.perf_counter()
    labels_sklearn = hdb.fit_predict(X)
    t1 = time.perf_counter()
    sklearn_time_ms = (t1 - t0) * 1000

    n_clusters_sklearn = len(set(labels_sklearn) - {-1})
    n_noise_sklearn = np.sum(labels_sklearn == -1)

    print(f"\nsklearn HDBSCAN results:")
    print(f"  Time:            {sklearn_time_ms:.1f} ms")
    print(f"  Clusters found:  {n_clusters_sklearn}")
    print(f"  Noise points:    {n_noise_sklearn} / {len(X)}")
    print(f"  ARI vs ground truth: {adjusted_rand_score(y_true, labels_sklearn):.4f}")
    print(f"  AMI vs ground truth: {adjusted_mutual_info_score(y_true, labels_sklearn):.4f}")

    # Save
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    data_path = os.path.join(args.out_dir, "hdbscan_bench.csv")
    labels_path = os.path.join(args.out_dir, "hdbscan_bench_labels.csv")
    summary_path = os.path.join(args.out_dir, "hdbscan_bench_summary.txt")

    np.savetxt(data_path, X, delimiter=",", fmt="%.10f")
    np.savetxt(labels_path, labels_sklearn, delimiter=",", fmt="%d")

    with open(summary_path, "w") as f:
        f.write(f"rows={X.shape[0]}\n")
        f.write(f"cols={X.shape[1]}\n")
        f.write(f"true_clusters={n_clusters_sklearn}\n")
        f.write(f"min_cluster_size={args.min_cluster_size}\n")
        f.write(f"min_samples={args.min_samples}\n")
        f.write(f"sklearn_clusters={n_clusters_sklearn}\n")
        f.write(f"sklearn_noise={n_noise_sklearn}\n")
        f.write(f"sklearn_time_ms={sklearn_time_ms:.1f}\n")
        f.write(f"ari={adjusted_rand_score(y_true, labels_sklearn):.6f}\n")
        f.write(f"ami={adjusted_mutual_info_score(y_true, labels_sklearn):.6f}\n")

    print(f"\nSaved:")
    print(f"  {data_path}   ({X.shape[0]} x {X.shape[1]})")
    print(f"  {labels_path}")
    print(f"  {summary_path}")

    # Plot the generated dataset with sklearn labels
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: ground truth
    plot_labels(axes[0], X[:, :2], y_true.astype(int), "Ground Truth Labels")

    # Right: sklearn HDBSCAN
    plot_labels(axes[1], X[:, :2], labels_sklearn,
                f"sklearn HDBSCAN ({n_clusters_sklearn} clusters, {n_noise_sklearn} noise, {sklearn_time_ms:.0f} ms)")

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "hdbscan_bench_data.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {plot_path}")

    print(f"\n--- To compare with oneDAL ---")
    print(f"1. Run oneDAL HDBSCAN bench on {data_path} with min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")
    print(f"2. Then run:  python scripts/hdbscan_compare.py --check")


def check():
    """Compare oneDAL output with sklearn reference."""
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    X = np.loadtxt("data/hdbscan_bench.csv", delimiter=",")
    labels_sklearn = np.loadtxt("data/hdbscan_bench_labels.csv", delimiter=",", dtype=int)

    # Try to load oneDAL output
    try:
        labels_onedal = np.loadtxt("data/hdbscan_bench_onedal_labels.csv", delimiter=",", dtype=int)
    except FileNotFoundError:
        print("ERROR: data/hdbscan_bench_onedal_labels.csv not found.")
        print("Run your oneDAL HDBSCAN bench example and save responses to that file.")
        return

    assert len(labels_sklearn) == len(labels_onedal), \
        f"Length mismatch: sklearn={len(labels_sklearn)}, onedal={len(labels_onedal)}"

    n_clusters_sk = len(set(labels_sklearn) - {-1})
    n_clusters_od = len(set(labels_onedal) - {-1})
    n_noise_sk = np.sum(labels_sklearn == -1)
    n_noise_od = np.sum(labels_onedal == -1)

    ari = adjusted_rand_score(labels_sklearn, labels_onedal)
    ami = adjusted_mutual_info_score(labels_sklearn, labels_onedal)
    exact_match = np.mean(labels_sklearn == labels_onedal)

    print("=" * 50)
    print("HDBSCAN Comparison: sklearn vs oneDAL")
    print("=" * 50)
    print(f"{'':>25} {'sklearn':>10} {'oneDAL':>10}")
    print(f"{'Clusters found':>25} {n_clusters_sk:>10} {n_clusters_od:>10}")
    print(f"{'Noise points':>25} {n_noise_sk:>10} {n_noise_od:>10}")
    print()
    print(f"  Adjusted Rand Index:          {ari:.4f}")
    print(f"  Adjusted Mutual Information:  {ami:.4f}")
    print(f"  Exact label match rate:       {exact_match:.4f}")
    print()

    if ari > 0.95:
        print("RESULT: Excellent agreement")
    elif ari > 0.8:
        print("RESULT: Good agreement")
    elif ari > 0.5:
        print("RESULT: Moderate agreement -- check edge cases")
    else:
        print("RESULT: Poor agreement -- likely implementation differences")

    # Per-cluster breakdown
    print("\nPer-cluster size comparison:")
    from collections import Counter
    sk_counts = Counter(labels_sklearn)
    od_counts = Counter(labels_onedal)
    all_labels = sorted(set(list(sk_counts.keys()) + list(od_counts.keys())))
    for lbl in all_labels:
        tag = "noise" if lbl == -1 else f"cluster {lbl}"
        print(f"  {tag:>15}: sklearn={sk_counts.get(lbl, 0):>5}, onedal={od_counts.get(lbl, 0):>5}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Top-left: sklearn labels
    plot_labels(axes[0, 0], X[:, :2], labels_sklearn,
                f"sklearn HDBSCAN ({n_clusters_sk} clusters, {n_noise_sk} noise)")

    # Top-right: oneDAL labels
    plot_labels(axes[0, 1], X[:, :2], labels_onedal,
                f"oneDAL HDBSCAN ({n_clusters_od} clusters, {n_noise_od} noise)")

    # Bottom-left: agreement/disagreement map
    agree = labels_sklearn == labels_onedal
    colors = np.full(len(X), 'green', dtype=object)
    colors[~agree] = 'red'
    both_noise = (labels_sklearn == -1) & (labels_onedal == -1)
    colors[both_noise] = 'lightgray'

    ax = axes[1, 0]
    for c, label in [('lightgray', 'Both noise'), ('green', 'Agree'), ('red', 'Disagree')]:
        mask = colors == c
        if mask.any():
            ax.scatter(X[mask, 0], X[mask, 1], c=c, s=8, alpha=0.7, label=label,
                       edgecolors='black' if c == 'red' else 'none',
                       linewidths=0.5 if c == 'red' else 0)
    ax.set_title(f"Agreement Map ({agree.sum()}/{len(X)} agree, {(~agree).sum()} disagree)",
                 fontsize=13)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.legend(fontsize=9)

    # Bottom-right: zoom into disagreement points
    ax = axes[1, 1]
    disagree_mask = ~agree
    if disagree_mask.any():
        ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=3, alpha=0.2)
        disagree_sk = labels_sklearn[disagree_mask]
        disagree_od = labels_onedal[disagree_mask]
        X_dis = X[disagree_mask]

        sk_noise = disagree_sk == -1
        od_noise = disagree_od == -1
        other = ~sk_noise & ~od_noise

        if sk_noise.any():
            ax.scatter(X_dis[sk_noise, 0], X_dis[sk_noise, 1], c='blue', s=25, marker='^',
                       alpha=0.8, label=f'sklearn=noise, oneDAL=cluster ({sk_noise.sum()})',
                       edgecolors='black', linewidths=0.5)
        if od_noise.any():
            ax.scatter(X_dis[od_noise, 0], X_dis[od_noise, 1], c='orange', s=25, marker='v',
                       alpha=0.8, label=f'sklearn=cluster, oneDAL=noise ({od_noise.sum()})',
                       edgecolors='black', linewidths=0.5)
        if other.any():
            ax.scatter(X_dis[other, 0], X_dis[other, 1], c='purple', s=25, marker='s',
                       alpha=0.8, label=f'Different clusters ({other.sum()})',
                       edgecolors='black', linewidths=0.5)

        ax.set_title(f"Disagreement Details ({disagree_mask.sum()} points)", fontsize=13)
        ax.legend(fontsize=9, loc='best')
    else:
        ax.text(0.5, 0.5, "Perfect agreement!\nNo disagreements.", ha='center', va='center',
                fontsize=16, transform=ax.transAxes)
        ax.set_title("Disagreement Details", fontsize=13)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')

    plt.tight_layout()
    plot_path = "data/hdbscan_bench_compare.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to: {plot_path}")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        check()
    else:
        main()
