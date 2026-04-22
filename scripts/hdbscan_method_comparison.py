#!/usr/bin/env python3
"""Compare HDBSCAN brute_force vs kd_tree methods across dataset sizes.
   Also compares against sklearn brute and kd_tree algorithms."""

import subprocess
import sys
import os
import time
import tempfile
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score


def generate_dataset(name, n, d):
    """Generate a dataset by name."""
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
    elif name == "uniform":
        X = rng.uniform(-10, 10, size=(n, d))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X


def run_oneDAL(example_bin, data_path, min_cluster_size, min_samples):
    """Run a oneDAL HDBSCAN example and parse results."""
    env = os.environ.copy()
    env["HDBSCAN_DATA_PATH"] = data_path
    env["HDBSCAN_MIN_CLUSTER_SIZE"] = str(min_cluster_size)
    env["HDBSCAN_MIN_SAMPLES"] = str(min_samples)
    env["ONEAPI_DEVICE_SELECTOR"] = "opencl:cpu"

    result = subprocess.run(
        [example_bin],
        capture_output=True, text=True, timeout=300, env=env
    )
    output = result.stdout + result.stderr

    time_ms = None
    labels = None
    n_clusters = None
    for line in output.strip().split("\n"):
        line = line.strip()
        if line.startswith("Time:"):
            time_ms = float(line.split(":")[1].strip().replace(" ms", ""))
        elif line.startswith("Cluster count:"):
            n_clusters = int(line.split(":")[1].strip())
        elif line.startswith("Labels:"):
            labels_str = line.replace("Labels: ", "").strip()
            labels = np.array([int(x) for x in labels_str.split()])

    return time_ms, n_clusters, labels


def run_sklearn(data, min_cluster_size, min_samples, algorithm="brute"):
    """Run sklearn HDBSCAN and return time + labels."""
    hdb = SklearnHDBSCAN(algorithm=algorithm, min_cluster_size=min_cluster_size,
                          min_samples=min_samples)
    t0 = time.time()
    labels = hdb.fit_predict(data)
    t1 = time.time()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return (t1 - t0) * 1000, n_clusters, labels


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dal_root = os.path.dirname(base_dir)
    examples_dir = os.path.join(dal_root, "__release_lnx/daal/latest/examples/oneapi/dpc/_cmake_results/intel_intel64_so")
    bf_bin = os.path.join(examples_dir, "hdbscan_brute_force_batch")
    kd_bin = os.path.join(examples_dir, "hdbscan_kd_tree_batch")

    if not os.path.exists(bf_bin) or not os.path.exists(kd_bin):
        print(f"ERROR: Example binaries not found at {examples_dir}")
        sys.exit(1)

    # Test configurations: (dataset_name, N, D, min_cluster_size, min_samples)
    configs = [
        ("blobs", 500, 2, 10, 5),
        ("blobs", 1000, 2, 10, 5),
        ("moons", 2000, 2, 15, 10),
        ("blobs", 5000, 2, 15, 10),
        ("blobs", 5000, 5, 15, 10),
        ("blobs", 5000, 10, 15, 10),
        ("blobs", 10000, 2, 15, 10),
        ("blobs", 10000, 10, 15, 10),
        ("blobs", 20000, 2, 20, 10),
        ("blobs", 30000, 2, 20, 10),
        ("blobs", 50000, 2, 20, 10),
    ]

    # Header
    print(f"{'Dataset':<18} {'N':>6} {'D':>2} | "
          f"{'sk brute':>9} {'sk kd':>9} | "
          f"{'dal BF':>9} {'dal KD':>9} | "
          f"{'BF/sk_b':>8} {'KD/sk_kd':>9} | "
          f"{'ARI':>5}")
    print("-" * 120)

    for ds_name, n, d, mcs, ms in configs:
        dataset_label = f"{ds_name}_{n}_{d}d"

        data = generate_dataset(ds_name, n, d)

        # Save to temp CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmpfile = f.name
            for row in data:
                f.write(",".join(f"{v:.8f}" for v in row) + "\n")

        try:
            # sklearn brute
            sk_b_ms, sk_b_nc, sk_b_labels = run_sklearn(data, mcs, ms, "brute")

            # sklearn kd_tree
            if d <= 20:
                sk_k_ms, sk_k_nc, sk_k_labels = run_sklearn(data, mcs, ms, "kd_tree")
            else:
                sk_k_ms, sk_k_nc, sk_k_labels = None, None, None

            # oneDAL brute_force
            bf_ms, bf_nc, bf_labels = run_oneDAL(bf_bin, tmpfile, mcs, ms)

            # oneDAL kd_tree
            try:
                kd_ms, kd_nc, kd_labels = run_oneDAL(kd_bin, tmpfile, mcs, ms)
            except subprocess.TimeoutExpired:
                kd_ms, kd_nc, kd_labels = None, None, None

            # ARI: compare both oneDAL methods vs each other
            ari_bf_kd = adjusted_rand_score(bf_labels, kd_labels) if bf_labels is not None and kd_labels is not None else 0

            bf_speedup = sk_b_ms / bf_ms if bf_ms and bf_ms > 0 else 0
            kd_speedup = sk_k_ms / kd_ms if sk_k_ms and kd_ms and kd_ms > 0 else 0

            sk_k_str = f"{sk_k_ms:.0f}ms" if sk_k_ms else "N/A"
            kd_str = f"{kd_ms:.0f}ms" if kd_ms else "N/A"

            print(f"{dataset_label:<18} {n:>6} {d:>2} | "
                  f"{sk_b_ms:>7.0f}ms {sk_k_str:>9} | "
                  f"{bf_ms:>7.0f}ms {kd_str:>9} | "
                  f"{bf_speedup:>7.2f}x {kd_speedup:>8.2f}x | "
                  f"{ari_bf_kd:>5.3f}")

        finally:
            os.unlink(tmpfile)

    print()
    print("Legend:")
    print("  sk brute / sk kd = sklearn time with algorithm='brute' / 'kd_tree'")
    print("  dal BF / dal KD  = oneDAL time with method::brute_force / method::kd_tree")
    print("  BF/sk_b          = sklearn_brute / oneDAL_brute (speedup over sklearn brute)")
    print("  KD/sk_kd         = sklearn_kd / oneDAL_kd (speedup over sklearn kd_tree)")
    print("  ARI              = agreement between oneDAL brute_force and kd_tree (1.0 = identical)")


if __name__ == "__main__":
    main()
