#!/usr/bin/env python3
"""
Comprehensive HDBSCAN benchmark: oneDAL vs scikit-learn
Memory analysis + performance across dataset sizes + correctness validation.
"""

import subprocess
import sys
import os
import time
import tempfile
import warnings
import gc
import tracemalloc

warnings.filterwarnings('ignore')

import numpy as np
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAL_ROOT = os.path.dirname(BASE_DIR)
EXAMPLES_DIR = os.path.join(
    DAL_ROOT,
    "__release_lnx/daal/latest/examples/oneapi/dpc/_cmake_results/intel_intel64_so"
)
BF_BIN = os.path.join(EXAMPLES_DIR, "hdbscan_brute_force_batch")
KD_BIN = os.path.join(EXAMPLES_DIR, "hdbscan_kd_tree_batch")

# Dataset configurations: (name, N, D, min_cluster_size, min_samples)
CONFIGS = [
    # Small: warmup / baseline
    ("blobs", 500, 2, 10, 5),
    ("blobs", 1000, 2, 10, 5),
    ("blobs", 2000, 2, 15, 10),
    # Medium: sweet spot
    ("blobs", 3000, 2, 15, 10),
    ("blobs", 5000, 2, 15, 10),
    ("blobs", 5000, 5, 15, 10),
    ("blobs", 5000, 10, 15, 10),
    ("blobs", 5000, 20, 15, 10),
    ("moons", 5000, 2, 15, 10),
    ("blobs", 7500, 2, 15, 10),
    # Large: where performance matters
    ("blobs", 10000, 2, 15, 10),
    ("blobs", 10000, 5, 15, 10),
    ("blobs", 10000, 10, 15, 10),
    ("blobs", 10000, 20, 15, 10),
    ("blobs", 15000, 2, 20, 10),
    ("blobs", 20000, 2, 20, 10),
    ("blobs", 20000, 5, 20, 10),
    ("blobs", 20000, 10, 20, 10),
    # Very large: memory limits
    ("blobs", 30000, 2, 25, 15),
    ("blobs", 40000, 2, 25, 15),
    ("blobs", 50000, 2, 25, 15),
    ("blobs", 75000, 2, 30, 20),
    ("blobs", 100000, 2, 30, 20),
]


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
    elif name == "uniform":
        X = rng.uniform(-10, 10, size=(n, d))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X.astype(np.float32)


def save_csv(data, path):
    with open(path, "w") as f:
        for row in data:
            f.write(",".join(f"{v:.8f}" for v in row) + "\n")


def run_onedal(binary, data_path, mcs, ms, device="opencl:cpu", timeout=600):
    """Run oneDAL example, return (time_ms, n_clusters, labels, stdout)."""
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
    except subprocess.TimeoutExpired:
        return None, None, None, "TIMEOUT"
    except Exception as e:
        return None, None, None, f"ERROR: {e}"

    if result.returncode != 0:
        return None, None, None, f"EXIT {result.returncode}: {output[:200]}"

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
            labels_str = line.replace("Labels:", "").strip()
            labels = np.array([int(x) for x in labels_str.split()])

    return time_ms, n_clusters, labels, output


def run_sklearn(data, mcs, ms, algorithm="brute", timeout=600):
    """Run sklearn HDBSCAN, return (time_ms, n_clusters, labels, peak_mem_mb)."""
    gc.collect()
    tracemalloc.start()

    hdb = SklearnHDBSCAN(algorithm=algorithm, min_cluster_size=mcs,
                          min_samples=ms, copy=True)
    t0 = time.time()
    try:
        labels = hdb.fit_predict(data)
    except MemoryError:
        tracemalloc.stop()
        return None, None, None, None
    t1 = time.time()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return (t1 - t0) * 1000, n_clusters, labels, peak / 1024 / 1024


def theoretical_memory_mb(n, d, method, dtype_bytes=4):
    """Estimate peak memory in MB for oneDAL."""
    data_mb = n * d * dtype_bytes / 1024 / 1024
    if method == "brute_force":
        # N×N distance matrix + N×N MRD matrix + MST arrays
        dist_mb = n * n * dtype_bytes / 1024 / 1024
        mrd_mb = dist_mb
        mst_mb = n * 3 * 4 / 1024 / 1024  # from, to, weight arrays
        return data_mb + dist_mb + mrd_mb + mst_mb
    elif method == "kd_tree_cpu":
        # kd-tree nodes + k-NN arrays + MST arrays (no N² matrix)
        tree_mb = n * 24 / 1024 / 1024  # ~24 bytes per node
        knn_mb = n * 10 * dtype_bytes / 1024 / 1024  # k neighbors per point
        mst_mb = n * 3 * 4 / 1024 / 1024
        prim_mb = n * (dtype_bytes + 4 + 1) / 1024 / 1024  # min_edge, min_from, in_mst
        return data_mb + tree_mb + knn_mb + mst_mb + prim_mb
    elif method == "kd_tree_gpu":
        # Blocked: B×N distance block (~256MB cap) + core distances + host copy for Prim's
        block_mb = 256  # capped at 256MB
        core_mb = n * dtype_bytes / 1024 / 1024
        host_mb = data_mb + core_mb  # data + core on host for Prim's
        prim_mb = n * (dtype_bytes + 4 + 1) / 1024 / 1024
        return min(block_mb, n * n * dtype_bytes / 1024 / 1024) + core_mb + host_mb + prim_mb
    return 0


def fmt_time(ms):
    if ms is None:
        return "   N/A   "
    if ms < 1000:
        return f"{ms:7.0f}ms"
    else:
        return f"{ms/1000:6.1f}s "


def fmt_speedup(sk_ms, dal_ms):
    if sk_ms is None or dal_ms is None or dal_ms <= 0:
        return "   N/A "
    ratio = sk_ms / dal_ms
    return f"{ratio:6.2f}x"


def fmt_mem(mb):
    if mb is None:
        return "   N/A  "
    if mb < 1024:
        return f"{mb:6.0f}MB"
    else:
        return f"{mb/1024:5.1f}GB"


# ============================================================================
# Main benchmark
# ============================================================================

def main():
    if not os.path.exists(BF_BIN) or not os.path.exists(KD_BIN):
        print(f"ERROR: Binaries not found at {EXAMPLES_DIR}")
        sys.exit(1)

    print("=" * 140)
    print("HDBSCAN COMPREHENSIVE BENCHMARK: oneDAL vs scikit-learn")
    print("=" * 140)
    print()

    # ========================================================================
    # Part 1: Memory Analysis
    # ========================================================================
    print("=" * 100)
    print("PART 1: MEMORY ANALYSIS (theoretical peak + measured sklearn)")
    print("=" * 100)
    print()
    print(f"{'Dataset':<18} {'N':>6} {'D':>3} | "
          f"{'sk brute':>8} {'sk kd':>8} | "
          f"{'dal BF':>8} {'dal KD cpu':>10} {'dal KD gpu':>10} | "
          f"{'N^2*4B':>8}")
    print("-" * 100)

    mem_configs = [
        ("blobs", 1000, 2), ("blobs", 5000, 2), ("blobs", 10000, 2),
        ("blobs", 20000, 2), ("blobs", 30000, 2), ("blobs", 50000, 2),
        ("blobs", 75000, 2), ("blobs", 100000, 2),
        ("blobs", 10000, 10), ("blobs", 10000, 20), ("blobs", 10000, 50),
    ]

    for ds_name, n, d in mem_configs:
        label = f"{ds_name}_{n}_{d}d"
        n2_mb = n * n * 4 / 1024 / 1024

        # Measure sklearn memory (only for sizes that won't OOM)
        sk_b_mem = None
        sk_k_mem = None
        data = generate_dataset(ds_name, n, d)

        if n <= 30000:
            _, _, _, sk_b_mem = run_sklearn(data, 15, 10, "brute")
        if n <= 100000 and d <= 20:
            _, _, _, sk_k_mem = run_sklearn(data, 15, 10, "kd_tree")

        dal_bf_mem = theoretical_memory_mb(n, d, "brute_force")
        dal_kd_cpu_mem = theoretical_memory_mb(n, d, "kd_tree_cpu")
        dal_kd_gpu_mem = theoretical_memory_mb(n, d, "kd_tree_gpu")

        del data
        gc.collect()

        print(f"{label:<18} {n:>6} {d:>3} | "
              f"{fmt_mem(sk_b_mem)} {fmt_mem(sk_k_mem)} | "
              f"{fmt_mem(dal_bf_mem)} {fmt_mem(dal_kd_cpu_mem):>10} {fmt_mem(dal_kd_gpu_mem):>10} | "
              f"{fmt_mem(n2_mb)}")

    print()
    print("Notes:")
    print("  sk brute/kd    = measured Python peak memory (tracemalloc)")
    print("  dal BF/KD      = theoretical peak memory estimate (C++ allocations)")
    print("  N^2*4B         = raw float32 N×N matrix size (reference)")
    print("  oneDAL brute_force: O(N²) — allocates full distance + MRD matrices")
    print("  oneDAL kd_tree CPU: O(N*k + N*D) — no N² matrix, tree + k-NN heaps")
    print("  oneDAL kd_tree GPU: O(B*N) blocked — caps at ~256MB per block + host Prim's")
    print()

    # ========================================================================
    # Part 2: Memory limits — find the max N for each method
    # ========================================================================
    print("=" * 100)
    print("PART 2: MAXIMUM DATASET SIZE (find the wall for each method)")
    print("=" * 100)
    print()

    mem_limit_host = 128 * 1024  # 128 GB host RAM (approx)
    mem_limit_gpu = 48 * 1024  # 48 GB GPU memory (Max 1550)

    print(f"Assumed limits: Host RAM ~128 GB, GPU HBM ~48 GB")
    print()
    print(f"{'Method':<25} | {'Max N (2D)':>10} {'Mem@Max':>10} | {'Max N (10D)':>11} {'Mem@Max':>10} | Bottleneck")
    print("-" * 100)

    for method_name, method_key, mem_limit in [
        ("sklearn brute", "brute_force", mem_limit_host),
        ("sklearn kd_tree", "kd_tree_cpu", mem_limit_host),
        ("oneDAL brute_force CPU", "brute_force", mem_limit_host),
        ("oneDAL kd_tree CPU", "kd_tree_cpu", mem_limit_host),
        ("oneDAL brute_force GPU", "brute_force", mem_limit_gpu),
        ("oneDAL kd_tree GPU", "kd_tree_gpu", mem_limit_gpu),
    ]:
        max_n_2d = 0
        mem_2d = 0
        for n in range(1000, 2000001, 1000):
            m = theoretical_memory_mb(n, 2, method_key)
            if m > mem_limit:
                break
            max_n_2d = n
            mem_2d = m

        max_n_10d = 0
        mem_10d = 0
        for n in range(1000, 2000001, 1000):
            m = theoretical_memory_mb(n, 10, method_key)
            if m > mem_limit:
                break
            max_n_10d = n
            mem_10d = m

        if method_key == "brute_force":
            bottleneck = "O(N²) distance matrix"
        elif method_key == "kd_tree_cpu":
            bottleneck = "O(N*k) — practically unlimited"
        else:
            bottleneck = "O(B*N) blocked — 256MB cap per block"

        # sklearn brute uses ~2x because Python overhead
        if "sklearn brute" in method_name:
            bottleneck = "O(N²) distance + Python overhead (~2x)"
            max_n_2d = int(max_n_2d * 0.7)
            mem_2d *= 2
            max_n_10d = int(max_n_10d * 0.7)
            mem_10d *= 2

        print(f"{method_name:<25} | {max_n_2d:>10,} {fmt_mem(mem_2d)} | {max_n_10d:>11,} {fmt_mem(mem_10d)} | {bottleneck}")

    print()

    # ========================================================================
    # Part 3: Performance comparison — CPU
    # ========================================================================
    print("=" * 140)
    print("PART 3: PERFORMANCE COMPARISON — CPU (oneDAL vs sklearn)")
    print("=" * 140)
    print()
    print(f"{'Dataset':<18} {'N':>6} {'D':>3} | "
          f"{'sk brute':>9} {'sk kd':>9} | "
          f"{'dal BF':>9} {'dal KD':>9} | "
          f"{'BF vs sk_b':>10} {'KD vs sk_k':>10} | "
          f"{'Clust':>5} {'ARI bfkd':>8} {'ARI sk':>7}")
    print("-" * 140)

    results = []

    for ds_name, n, d, mcs, ms in CONFIGS:
        label = f"{ds_name}_{n}_{d}d"
        sys.stdout.write(f"\r  Running {label}...          ")
        sys.stdout.flush()

        data = generate_dataset(ds_name, n, d)

        # Save to CSV for oneDAL
        tmpfile = os.path.join(tempfile.gettempdir(), f"hdbscan_bench_{n}_{d}.csv")
        save_csv(data, tmpfile)

        # --- sklearn ---
        sk_b_ms, sk_b_nc, sk_b_labels, _ = (None, None, None, None)
        sk_k_ms, sk_k_nc, sk_k_labels, _ = (None, None, None, None)

        if n <= 50000:
            sk_b_ms, sk_b_nc, sk_b_labels, _ = run_sklearn(data, mcs, ms, "brute")
        if n <= 100000 and d <= 20:
            sk_k_ms, sk_k_nc, sk_k_labels, _ = run_sklearn(data, mcs, ms, "kd_tree")

        # --- oneDAL CPU brute_force ---
        bf_ms, bf_nc, bf_labels, bf_out = (None, None, None, None)
        if n <= 50000:
            bf_ms, bf_nc, bf_labels, bf_out = run_onedal(BF_BIN, tmpfile, mcs, ms, "opencl:cpu")

        # --- oneDAL CPU kd_tree ---
        kd_ms, kd_nc, kd_labels, kd_out = run_onedal(KD_BIN, tmpfile, mcs, ms, "opencl:cpu")

        # ARI between methods
        ari_bf_kd = None
        if bf_labels is not None and kd_labels is not None and len(bf_labels) == len(kd_labels):
            ari_bf_kd = adjusted_rand_score(bf_labels, kd_labels)

        # ARI vs sklearn
        ref_labels = sk_b_labels if sk_b_labels is not None else sk_k_labels
        dal_labels = bf_labels if bf_labels is not None else kd_labels
        ari_sk = None
        if ref_labels is not None and dal_labels is not None and len(ref_labels) == len(dal_labels):
            ari_sk = adjusted_rand_score(ref_labels, dal_labels)

        nc_display = bf_nc or kd_nc or 0

        bf_speedup = fmt_speedup(sk_b_ms, bf_ms)
        kd_speedup = fmt_speedup(sk_k_ms, kd_ms)

        sys.stdout.write(f"\r")
        print(f"{label:<18} {n:>6} {d:>3} | "
              f"{fmt_time(sk_b_ms)} {fmt_time(sk_k_ms)} | "
              f"{fmt_time(bf_ms)} {fmt_time(kd_ms)} | "
              f"{bf_speedup:>10} {kd_speedup:>10} | "
              f"{nc_display:>5} "
              f"{'  N/A  ' if ari_bf_kd is None else f'{ari_bf_kd:>7.4f}'} "
              f"{'  N/A' if ari_sk is None else f'{ari_sk:>6.4f}'}")

        results.append({
            'label': label, 'n': n, 'd': d, 'mcs': mcs, 'ms': ms,
            'sk_b_ms': sk_b_ms, 'sk_k_ms': sk_k_ms,
            'bf_ms': bf_ms, 'kd_ms': kd_ms,
            'bf_nc': bf_nc, 'kd_nc': kd_nc,
            'ari_bf_kd': ari_bf_kd, 'ari_sk': ari_sk,
        })

        try:
            os.unlink(tmpfile)
        except:
            pass
        del data
        gc.collect()

    print()

    # ========================================================================
    # Part 4: Performance comparison — GPU (where binaries run on GPU)
    # ========================================================================
    print("=" * 140)
    print("PART 4: PERFORMANCE COMPARISON — GPU (oneDAL brute_force vs kd_tree)")
    print("=" * 140)
    print()

    # Only run GPU on sizes that fit in GPU memory
    gpu_configs = [c for c in CONFIGS if c[1] <= 20000]

    print(f"{'Dataset':<18} {'N':>6} {'D':>3} | "
          f"{'GPU BF':>9} {'GPU KD':>9} | "
          f"{'CPU BF':>9} {'CPU KD':>9} | "
          f"{'sk brute':>9} | "
          f"{'GPU BF/sk':>9} {'GPU KD/sk':>9}")
    print("-" * 140)

    for ds_name, n, d, mcs, ms in gpu_configs:
        label = f"{ds_name}_{n}_{d}d"
        sys.stdout.write(f"\r  Running {label} on GPU...          ")
        sys.stdout.flush()

        data = generate_dataset(ds_name, n, d)
        tmpfile = os.path.join(tempfile.gettempdir(), f"hdbscan_bench_{n}_{d}.csv")
        save_csv(data, tmpfile)

        # GPU brute_force
        gpu_bf_ms, _, _, _ = run_onedal(BF_BIN, tmpfile, mcs, ms, "level_zero:gpu")
        # GPU kd_tree
        gpu_kd_ms, _, _, _ = run_onedal(KD_BIN, tmpfile, mcs, ms, "level_zero:gpu")

        # Get matching CPU results from Part 3
        r = next((r for r in results if r['label'] == label), None)
        cpu_bf_ms = r['bf_ms'] if r else None
        cpu_kd_ms = r['kd_ms'] if r else None
        sk_b_ms = r['sk_b_ms'] if r else None

        sys.stdout.write(f"\r")
        print(f"{label:<18} {n:>6} {d:>3} | "
              f"{fmt_time(gpu_bf_ms)} {fmt_time(gpu_kd_ms)} | "
              f"{fmt_time(cpu_bf_ms)} {fmt_time(cpu_kd_ms)} | "
              f"{fmt_time(sk_b_ms)} | "
              f"{fmt_speedup(sk_b_ms, gpu_bf_ms):>9} {fmt_speedup(sk_b_ms, gpu_kd_ms):>9}")

        try:
            os.unlink(tmpfile)
        except:
            pass
        del data
        gc.collect()

    print()

    # ========================================================================
    # Part 5: Dimensionality scaling
    # ========================================================================
    print("=" * 100)
    print("PART 5: DIMENSIONALITY SCALING (N=10000, D varies)")
    print("=" * 100)
    print()

    dim_configs = [
        (10000, 2), (10000, 5), (10000, 10), (10000, 20),
        (10000, 50), (10000, 100),
    ]

    print(f"{'D':>4} | {'sk brute':>9} {'sk kd':>9} | "
          f"{'dal BF':>9} {'dal KD':>9} | "
          f"{'BF spdup':>9} {'KD spdup':>9} | "
          f"{'ARI':>5}")
    print("-" * 90)

    for n, d in dim_configs:
        label = f"blobs_10k_{d}d"
        sys.stdout.write(f"\r  Running D={d}...          ")
        sys.stdout.flush()

        data = generate_dataset("blobs", n, d)
        tmpfile = os.path.join(tempfile.gettempdir(), f"hdbscan_bench_{n}_{d}.csv")
        save_csv(data, tmpfile)

        sk_b_ms, _, sk_b_labels, _ = run_sklearn(data, 15, 10, "brute")
        sk_k_ms, _, _, _ = (None, None, None, None)
        if d <= 20:
            sk_k_ms, _, _, _ = run_sklearn(data, 15, 10, "kd_tree")

        bf_ms, _, bf_labels, _ = run_onedal(BF_BIN, tmpfile, 15, 10, "opencl:cpu")
        kd_ms, _, kd_labels, _ = run_onedal(KD_BIN, tmpfile, 15, 10, "opencl:cpu")

        dal_labels = bf_labels if bf_labels is not None else kd_labels
        ari = None
        if sk_b_labels is not None and dal_labels is not None:
            ari = adjusted_rand_score(sk_b_labels, dal_labels)

        sys.stdout.write(f"\r")
        print(f"{d:>4} | {fmt_time(sk_b_ms)} {fmt_time(sk_k_ms)} | "
              f"{fmt_time(bf_ms)} {fmt_time(kd_ms)} | "
              f"{fmt_speedup(sk_b_ms, bf_ms):>9} {fmt_speedup(sk_k_ms, kd_ms):>9} | "
              f"{'  N/A' if ari is None else f'{ari:>5.3f}'}")

        try:
            os.unlink(tmpfile)
        except:
            pass
        del data
        gc.collect()

    print()

    # ========================================================================
    # Part 6: Summary and Recommendations
    # ========================================================================
    print("=" * 100)
    print("PART 6: SUMMARY")
    print("=" * 100)
    print()

    # Compute average speedups
    bf_speedups = []
    kd_speedups = []
    for r in results:
        if r['sk_b_ms'] and r['bf_ms'] and r['bf_ms'] > 0:
            bf_speedups.append(r['sk_b_ms'] / r['bf_ms'])
        if r['sk_k_ms'] and r['kd_ms'] and r['kd_ms'] > 0:
            kd_speedups.append(r['sk_k_ms'] / r['kd_ms'])

    if bf_speedups:
        print(f"  oneDAL brute_force vs sklearn brute:")
        print(f"    Average speedup: {np.mean(bf_speedups):.2f}x")
        print(f"    Min speedup:     {np.min(bf_speedups):.2f}x")
        print(f"    Max speedup:     {np.max(bf_speedups):.2f}x")
        print(f"    Datasets tested: {len(bf_speedups)}")
    print()
    if kd_speedups:
        print(f"  oneDAL kd_tree vs sklearn kd_tree:")
        print(f"    Average speedup: {np.mean(kd_speedups):.2f}x")
        print(f"    Min speedup:     {np.min(kd_speedups):.2f}x")
        print(f"    Max speedup:     {np.max(kd_speedups):.2f}x")
        print(f"    Datasets tested: {len(kd_speedups)}")
    print()

    # Correctness summary
    aris = [r['ari_sk'] for r in results if r['ari_sk'] is not None]
    if aris:
        print(f"  Correctness (ARI vs sklearn):")
        print(f"    Mean ARI:  {np.mean(aris):.4f}")
        print(f"    Min ARI:   {np.min(aris):.4f}")
        print(f"    All 1.0:   {'YES' if all(a >= 0.999 for a in aris) else 'NO'}")
        print(f"    Datasets:  {len(aris)}")
    print()

    print("  Memory limits (theoretical, float32):")
    print(f"    brute_force: max ~{int(np.sqrt(128 * 1024 * 1024 * 1024 / 4 / 2)):,} points (128GB RAM, 2×N² matrices)")
    print(f"    kd_tree CPU: practically unlimited (O(N*k) memory)")
    print(f"    kd_tree GPU: limited by 256MB block + host RAM for Prim's")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
