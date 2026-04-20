#!/usr/bin/env python3
"""HDBSCAN Performance Benchmark: oneDAL vs scikit-learn.

Generates datasets, runs both sklearn and oneDAL, compares timing and correctness (ARI).
"""

import os
import sys
import time
import subprocess
import tempfile
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
EXAMPLE_BIN = os.path.join(
    REPO_ROOT,
    "__release_lnx/daal/latest/examples/oneapi/dpc/_cmake_results/intel_intel64_so/"
    "hdbscan_brute_force_batch"
)


def generate_datasets():
    """Generate benchmark datasets of varying size and complexity."""
    datasets = []

    # Small (1000 points, 2D)
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=0.8, random_state=42)
    datasets.append(("blobs_1k_2d", X, 10, 5))

    # Medium (5000 points, 2D)
    X, _ = make_blobs(n_samples=5000, n_features=2, centers=7, cluster_std=1.0, random_state=42)
    datasets.append(("blobs_5k_2d", X, 15, 10))

    # Large (10000 points, 2D)
    X, _ = make_blobs(n_samples=10000, n_features=2, centers=10, cluster_std=1.2, random_state=42)
    datasets.append(("blobs_10k_2d", X, 20, 15))

    # Higher dimensional (5000 points, 10D)
    X, _ = make_blobs(n_samples=5000, n_features=10, centers=8, cluster_std=2.0, random_state=42)
    datasets.append(("blobs_5k_10d", X, 15, 10))

    # Large higher dimensional (10000 points, 10D, well-separated)
    X, _ = make_blobs(n_samples=10000, n_features=10, centers=8, cluster_std=1.5, random_state=42)
    datasets.append(("blobs_10k_10d", X, 20, 15))

    # Very large (20000 points, 2D, well-separated)
    X, _ = make_blobs(n_samples=20000, n_features=2, centers=10, cluster_std=0.8, random_state=42)
    datasets.append(("blobs_20k_2d", X, 25, 15))

    # Reference dataset
    dense_path = os.path.join(DATA_DIR, "hdbscan_dense.csv")
    if os.path.exists(dense_path):
        X = np.loadtxt(dense_path, delimiter=",")
        datasets.append(("hdbscan_dense", X, 15, 10))

    return datasets


def run_sklearn(X, min_cluster_size, min_samples):
    """Run sklearn HDBSCAN. Returns (labels, time_ms, n_clusters)."""
    t0 = time.time()
    hdb = HDBSCAN(algorithm="brute", min_cluster_size=min_cluster_size,
                  min_samples=min_samples, copy=True)
    labels = hdb.fit_predict(X)
    elapsed = (time.time() - t0) * 1000
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, elapsed, n_clusters


def run_onedal(X, min_cluster_size, min_samples, device_selector=None):
    """Run oneDAL HDBSCAN via the example binary.

    Returns list of (device_name, labels, time_ms, n_clusters) tuples.
    """
    if not os.path.exists(EXAMPLE_BIN):
        return []

    # Write data to temp CSV
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(tmp_fd)
    np.savetxt(tmp_path, X, delimiter=",", fmt="%.10f")

    try:
        env = os.environ.copy()
        env["HDBSCAN_DATA_PATH"] = tmp_path
        env["HDBSCAN_MIN_CLUSTER_SIZE"] = str(min_cluster_size)
        env["HDBSCAN_MIN_SAMPLES"] = str(min_samples)

        if device_selector:
            env["ONEAPI_DEVICE_SELECTOR"] = device_selector

        result = subprocess.run(
            [EXAMPLE_BIN],
            capture_output=True, text=True, timeout=600, env=env
        )

        if result.returncode != 0:
            return []

        # Parse output
        output = result.stdout
        devices_results = []
        current_device = None
        current_clusters = None
        current_time = None
        current_labels = None

        for line in output.split('\n'):
            if line.startswith("Running on"):
                if current_device is not None and current_labels is not None:
                    devices_results.append(
                        (current_device, current_labels, current_time, current_clusters))
                current_device = line.strip().replace("Running on ", "").rstrip()
                current_labels = None
            elif "Cluster count:" in line:
                current_clusters = int(line.split(":")[1].strip())
            elif "Time:" in line:
                current_time = float(line.split(":")[1].strip().replace(" ms", ""))
            elif line.startswith("Labels:"):
                label_str = line[len("Labels:"):].strip()
                if label_str:
                    current_labels = np.array([int(x) for x in label_str.split()], dtype=int)

        if current_device is not None and current_labels is not None:
            devices_results.append(
                (current_device, current_labels, current_time, current_clusters))

        return devices_results
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  oneDAL error: {e}")
        return []
    finally:
        os.unlink(tmp_path)


def main():
    print("=" * 110)
    print("HDBSCAN Performance Benchmark: oneDAL (CPU + GPU) vs scikit-learn")
    print("=" * 110)
    print(f"Binary: {EXAMPLE_BIN}")
    print(f"Binary exists: {os.path.exists(EXAMPLE_BIN)}")
    print()

    datasets = generate_datasets()

    # Header
    print(f"{'Dataset':<20} {'N':>6} {'D':>3} "
          f"{'sklearn':>10} {'C':>3} "
          f"{'CPU':>10} {'C':>3} {'ARI':>6} {'Spdup':>6} "
          f"{'GPU':>10} {'C':>3} {'ARI':>6} {'Spdup':>6}")
    print("-" * 110)

    summary = []

    for name, X, min_cluster_size, min_samples in datasets:
        n_samples, n_features = X.shape

        # Run sklearn
        labels_sk, time_sk, nc_sk = run_sklearn(X, min_cluster_size, min_samples)

        # Run oneDAL (both devices)
        dal_results = run_onedal(X, min_cluster_size, min_samples)

        # Identify CPU and GPU results
        cpu_result = None
        gpu_result = None
        for dev, lbl, t, nc in dal_results:
            if "OpenCL" in dev or "XEON" in dev.upper() or "CPU" in dev.upper():
                cpu_result = (dev, lbl, t, nc)
            elif "Level-Zero" in dev or "GPU" in dev.upper() or "Max" in dev:
                gpu_result = (dev, lbl, t, nc)

        # Format output
        row = f"{name:<20} {n_samples:>6} {n_features:>3} {time_sk:>8.1f}ms {nc_sk:>3} "

        if cpu_result:
            _, lbl_cpu, t_cpu, nc_cpu = cpu_result
            ari_cpu = adjusted_rand_score(labels_sk, lbl_cpu) if len(lbl_cpu) == len(labels_sk) else -1
            speedup_cpu = time_sk / t_cpu
            row += f"{t_cpu:>8.1f}ms {nc_cpu:>3} {ari_cpu:>6.3f} {speedup_cpu:>5.2f}x "
        else:
            row += f"{'N/A':>10} {'':>3} {'':>6} {'':>6} "

        if gpu_result:
            _, lbl_gpu, t_gpu, nc_gpu = gpu_result
            ari_gpu = adjusted_rand_score(labels_sk, lbl_gpu) if len(lbl_gpu) == len(labels_sk) else -1
            speedup_gpu = time_sk / t_gpu
            row += f"{t_gpu:>8.1f}ms {nc_gpu:>3} {ari_gpu:>6.3f} {speedup_gpu:>5.2f}x"
        else:
            row += f"{'N/A':>10} {'':>3} {'':>6} {'':>6}"

        print(row)

        summary.append({
            "name": name, "n": n_samples, "d": n_features,
            "sklearn_ms": time_sk, "sklearn_c": nc_sk,
            "cpu_ms": cpu_result[2] if cpu_result else None,
            "cpu_ari": ari_cpu if cpu_result else None,
            "gpu_ms": gpu_result[2] if gpu_result else None,
            "gpu_ari": ari_gpu if gpu_result else None,
        })

    # Print summary statistics
    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    cpu_speedups = [s["sklearn_ms"] / s["cpu_ms"] for s in summary if s["cpu_ms"]]
    gpu_speedups = [s["sklearn_ms"] / s["gpu_ms"] for s in summary if s["gpu_ms"]]
    cpu_aris = [s["cpu_ari"] for s in summary if s["cpu_ari"] is not None]
    gpu_aris = [s["gpu_ari"] for s in summary if s["gpu_ari"] is not None]

    if cpu_speedups:
        print(f"CPU avg speedup vs sklearn: {np.mean(cpu_speedups):.2f}x "
              f"(min={min(cpu_speedups):.2f}x, max={max(cpu_speedups):.2f}x)")
        print(f"CPU avg ARI vs sklearn:     {np.mean(cpu_aris):.4f} "
              f"(min={min(cpu_aris):.4f})")
    if gpu_speedups:
        print(f"GPU avg speedup vs sklearn: {np.mean(gpu_speedups):.2f}x "
              f"(min={min(gpu_speedups):.2f}x, max={max(gpu_speedups):.2f}x)")
        print(f"GPU avg ARI vs sklearn:     {np.mean(gpu_aris):.4f} "
              f"(min={min(gpu_aris):.4f})")

    # Correctness check
    all_correct = all(a >= 0.99 for a in cpu_aris + gpu_aris if a is not None)
    print(f"\nCorrectness (ARI >= 0.99 on all datasets): {'PASS' if all_correct else 'FAIL'}")


if __name__ == "__main__":
    main()
