#!/usr/bin/env python3
"""
Benchmark: sklearnex HDBSCAN (oneDAL backend) vs stock sklearn HDBSCAN.

Measures wall-clock time, ARI vs ground truth, ARI between implementations,
and cluster counts across multiple dataset sizes and algorithm choices.

Usage:
    # Ensure sklearnex is built against patched oneDAL:
    #   conda activate build_sklearnex
    #   source __release_lnx/daal/latest/env/vars.sh
    #   cd ../scikit-learn-intelex && python setup.py develop --no-deps
    #
    # Then run:
    python scripts/benchmark_sklearnex_hdbscan.py
"""

import time
import sys
import warnings

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

try:
    from sklearnex.cluster import HDBSCAN as SklearnexHDBSCAN
except ImportError:
    print("ERROR: sklearnex not available. Build it first:")
    print("  conda activate build_sklearnex")
    print("  cd scikit-learn-intelex && python setup.py develop --no-deps")
    sys.exit(1)

# Suppress sklearn convergence/future warnings
warnings.filterwarnings("ignore")

# ---- Configuration ----
SIZES = [1000, 5000, 10000, 20000, 50000]
N_FEATURES = 10
N_CENTERS = 10
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 5
N_RUNS = 3  # take median
ALGORITHMS = ["brute", "auto"]  # auto maps to kd_tree for euclidean in sklearnex
RANDOM_STATE = 42


def count_clusters(labels):
    return len(set(labels)) - (1 if -1 in labels else 0)


def noise_fraction(labels):
    return np.sum(labels == -1) / len(labels)


def bench_one(X, y_true, algorithm):
    """Benchmark sklearn and sklearnex on a single dataset."""

    # sklearn
    times_sk = []
    sk_labels = None
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sk = SklearnHDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            algorithm=algorithm,
        )
        sk.fit(X)
        times_sk.append(time.perf_counter() - t0)
        sk_labels = sk.labels_

    # sklearnex
    times_sx = []
    sx_labels = None
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sx = SklearnexHDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            algorithm=algorithm,
        )
        sx.fit(X)
        times_sx.append(time.perf_counter() - t0)
        sx_labels = sx.labels_

    t_sk = np.median(times_sk)
    t_sx = np.median(times_sx)
    speedup = t_sk / t_sx if t_sx > 0 else float("inf")

    ari_sk_true = adjusted_rand_score(y_true, sk_labels)
    ari_sx_true = adjusted_rand_score(y_true, sx_labels)
    ari_cross = adjusted_rand_score(sk_labels, sx_labels)

    nc_sk = count_clusters(sk_labels)
    nc_sx = count_clusters(sx_labels)
    nf_sk = noise_fraction(sk_labels)
    nf_sx = noise_fraction(sx_labels)

    return {
        "t_sk": t_sk,
        "t_sx": t_sx,
        "speedup": speedup,
        "ari_sk_true": ari_sk_true,
        "ari_sx_true": ari_sx_true,
        "ari_cross": ari_cross,
        "nc_sk": nc_sk,
        "nc_sx": nc_sx,
        "nf_sk": nf_sk,
        "nf_sx": nf_sx,
    }


def main():
    print("=" * 110)
    print("HDBSCAN Benchmark: sklearnex (oneDAL) vs sklearn")
    print(f"Config: features={N_FEATURES}, centers={N_CENTERS}, "
          f"mcs={MIN_CLUSTER_SIZE}, ms={MIN_SAMPLES}, runs={N_RUNS}")
    print("=" * 110)

    for algorithm in ALGORITHMS:
        print(f"\n--- algorithm='{algorithm}' ---")
        print(f"{'N':>7s} | {'sklearn':>9s} {'sklearnex':>9s} {'speedup':>8s} | "
              f"{'ARI(sk)':>7s} {'ARI(sx)':>7s} {'ARI(x)':>7s} | "
              f"{'cl_sk':>5s} {'cl_sx':>5s} {'noise_sk':>8s} {'noise_sx':>8s}")
        print("-" * 110)

        # Warmup run
        X_warm, _ = make_blobs(n_samples=200, n_features=N_FEATURES,
                               centers=3, random_state=0)
        SklearnHDBSCAN(min_cluster_size=5).fit(X_warm)
        SklearnexHDBSCAN(min_cluster_size=5).fit(X_warm)

        for n in SIZES:
            X, y_true = make_blobs(
                n_samples=n,
                n_features=N_FEATURES,
                centers=N_CENTERS,
                cluster_std=1.0,
                random_state=RANDOM_STATE,
            )

            r = bench_one(X, y_true, algorithm)

            print(
                f"{n:7d} | "
                f"{r['t_sk']:9.3f}s {r['t_sx']:9.3f}s {r['speedup']:7.1f}x | "
                f"{r['ari_sk_true']:7.4f} {r['ari_sx_true']:7.4f} {r['ari_cross']:7.4f} | "
                f"{r['nc_sk']:5d} {r['nc_sx']:5d} {r['nf_sk']:8.2%} {r['nf_sx']:8.2%}"
            )

    print("\n" + "=" * 110)
    print("Legend:")
    print("  sklearn    = stock sklearn.cluster.HDBSCAN (median of 3 runs)")
    print("  sklearnex  = sklearnex.cluster.HDBSCAN backed by oneDAL (median of 3 runs)")
    print("  speedup    = sklearn_time / sklearnex_time")
    print("  ARI(sk)    = Adjusted Rand Index of sklearn labels vs ground truth")
    print("  ARI(sx)    = Adjusted Rand Index of sklearnex labels vs ground truth")
    print("  ARI(x)     = Adjusted Rand Index between sklearn and sklearnex labels")
    print("  cl_sk/sx   = number of clusters found")
    print("  noise_sk/sx = fraction of points labeled as noise")
    print("=" * 110)


if __name__ == "__main__":
    main()
