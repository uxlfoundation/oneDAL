#!/usr/bin/env python3
"""
Hard benchmark: sklearnex vs sklearn HDBSCAN on challenging datasets.
Tests overlapping clusters, noise, varying density, and high dimensions.
"""

import time
import sys
import warnings

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

try:
    from sklearnex.cluster import HDBSCAN as SklearnexHDBSCAN
except ImportError:
    print("ERROR: sklearnex not available")
    sys.exit(1)

warnings.filterwarnings("ignore")

N_RUNS = 3


def count_clusters(labels):
    return len(set(labels)) - (1 if -1 in labels else 0)


def noise_fraction(labels):
    return np.sum(labels == -1) / len(labels)


def bench_one(X, y_true, algorithm, mcs, ms):
    times_sk, times_sx = [], []
    sk_labels = sx_labels = None

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sk = SklearnHDBSCAN(min_cluster_size=mcs, min_samples=ms, algorithm=algorithm)
        sk.fit(X)
        times_sk.append(time.perf_counter() - t0)
        sk_labels = sk.labels_

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sx = SklearnexHDBSCAN(min_cluster_size=mcs, min_samples=ms, algorithm=algorithm)
        sx.fit(X)
        times_sx.append(time.perf_counter() - t0)
        sx_labels = sx.labels_

    t_sk = np.median(times_sk)
    t_sx = np.median(times_sx)
    speedup = t_sk / t_sx if t_sx > 0 else float("inf")
    ari_cross = adjusted_rand_score(sk_labels, sx_labels)
    ari_sk = adjusted_rand_score(y_true, sk_labels) if y_true is not None else -1
    ari_sx = adjusted_rand_score(y_true, sx_labels) if y_true is not None else -1

    return {
        "t_sk": t_sk, "t_sx": t_sx, "speedup": speedup,
        "ari_sk": ari_sk, "ari_sx": ari_sx, "ari_cross": ari_cross,
        "nc_sk": count_clusters(sk_labels), "nc_sx": count_clusters(sx_labels),
        "nf_sk": noise_fraction(sk_labels), "nf_sx": noise_fraction(sx_labels),
    }


def make_overlapping(n, dims, centers, std, noise_frac, seed):
    rng = np.random.RandomState(seed)
    n_signal = int(n * (1 - noise_frac))
    X_signal, y_signal = make_blobs(
        n_samples=n_signal, n_features=dims, centers=centers,
        cluster_std=std, random_state=seed
    )
    n_noise = n - n_signal
    X_noise = rng.uniform(
        X_signal.min(axis=0) - 2 * std,
        X_signal.max(axis=0) + 2 * std,
        size=(n_noise, dims)
    )
    X = np.vstack([X_signal, X_noise])
    y = np.concatenate([y_signal, np.full(n_noise, -1)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


datasets = [
    # (name, X, y_true, algorithm, mcs, ms)
    ("well_sep_10k", *make_blobs(n_samples=10000, n_features=10, centers=10, cluster_std=0.5, random_state=42), "auto", 15, 5),
    ("overlap_10k", *make_overlapping(10000, 10, 5, 3.0, 0.05, 42), "auto", 20, 10),
    ("overlap_20k", *make_overlapping(20000, 10, 8, 2.5, 0.1, 42), "auto", 25, 10),
    ("high_dim_10k", *make_blobs(n_samples=10000, n_features=50, centers=5, cluster_std=2.0, random_state=42), "auto", 20, 10),
    ("high_dim_50k", *make_blobs(n_samples=50000, n_features=50, centers=10, cluster_std=2.0, random_state=42), "auto", 30, 15),
    ("overlap_10k_brute", *make_overlapping(10000, 10, 5, 3.0, 0.05, 42), "brute", 20, 10),
    ("overlap_20k_brute", *make_overlapping(20000, 10, 8, 2.5, 0.1, 42), "brute", 25, 10),
    ("varied_density", *make_overlapping(15000, 5, 7, 1.5, 0.08, 123), "auto", 15, 5),
]


def main():
    print("=" * 120)
    print("HDBSCAN Hard Benchmark: sklearnex (oneDAL) vs sklearn — challenging datasets")
    print("=" * 120)
    print(f"{'Dataset':<22s} {'algo':>6s} | {'sklearn':>9s} {'sklearnex':>9s} {'speedup':>8s} | "
          f"{'ARI(sk)':>7s} {'ARI(sx)':>7s} {'ARI(x)':>7s} | "
          f"{'cl_sk':>5s} {'cl_sx':>5s} {'noise_sk':>8s} {'noise_sx':>8s}")
    print("-" * 120)

    # Warmup
    X_w, _ = make_blobs(n_samples=200, n_features=5, centers=3, random_state=0)
    SklearnHDBSCAN(min_cluster_size=5).fit(X_w)
    SklearnexHDBSCAN(min_cluster_size=5).fit(X_w)

    for name, X, y_true, algo, mcs, ms in datasets:
        r = bench_one(X, y_true, algo, mcs, ms)
        print(
            f"{name:<22s} {algo:>6s} | "
            f"{r['t_sk']:9.3f}s {r['t_sx']:9.3f}s {r['speedup']:7.1f}x | "
            f"{r['ari_sk']:7.4f} {r['ari_sx']:7.4f} {r['ari_cross']:7.4f} | "
            f"{r['nc_sk']:5d} {r['nc_sx']:5d} {r['nf_sk']:8.2%} {r['nf_sx']:8.2%}"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
