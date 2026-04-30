#!/usr/bin/env python3
"""
HDBSCAN Performance Benchmark: stock sklearn vs oneDAL-accelerated (via sklearnex)

Tests across multiple dataset sizes, dimensions, cluster structures, and metrics.
Measures wall-clock time, correctness (ARI vs sklearn), and speedup factors.

Usage:
    python benchmark_hdbscan.py
"""

import sys
import time
import warnings
import numpy as np

# Force unbuffered output for real-time progress
sys.stdout.reconfigure(line_buffering=True)
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

# Stock sklearn (bypass any patching)
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN as sklearn_HDBSCAN

# oneDAL direct (fastest path, no dispatch overhead)
from onedal.cluster import HDBSCAN as onedal_HDBSCAN

warnings.filterwarnings("ignore")


# ── Dataset generators ────────────────────────────────────────────────────────

def make_dataset(name, n_samples, n_features=2, random_state=42):
    """Generate a dataset by name."""
    if name == "blobs":
        n_centers = max(3, min(20, n_samples // 2000))
        X, y = make_blobs(
            n_samples=n_samples, n_features=n_features,
            centers=n_centers, cluster_std=1.0, random_state=random_state
        )
    elif name == "varied_density":
        rng = np.random.RandomState(random_state)
        n1, n2, n3 = n_samples // 3, n_samples // 3, n_samples - 2 * (n_samples // 3)
        c1 = rng.randn(n1, n_features) * 0.3
        c2 = rng.randn(n2, n_features) * 1.5 + np.array([10] + [0] * (n_features - 1))
        c3 = rng.randn(n3, n_features) * 0.8 + np.array([5, 8] + [0] * (n_features - 2))
        X = np.vstack([c1, c2, c3])
        y = np.concatenate([np.zeros(n1), np.ones(n2), np.full(n3, 2)]).astype(int)
    elif name == "noisy":
        rng = np.random.RandomState(random_state)
        n_clean = int(n_samples * 0.9)
        n_noise = n_samples - n_clean
        X_clean, y_clean = make_blobs(
            n_samples=n_clean, n_features=n_features,
            centers=5, cluster_std=0.8, random_state=random_state
        )
        X_noise = rng.uniform(X_clean.min(axis=0) - 5, X_clean.max(axis=0) + 5,
                              size=(n_noise, n_features))
        X = np.vstack([X_clean, X_noise])
        y = np.concatenate([y_clean, np.full(n_noise, -1)]).astype(int)
    elif name == "high_dim":
        n_informative = min(10, n_features)
        X_info, y = make_blobs(
            n_samples=n_samples, n_features=n_informative,
            centers=4, cluster_std=1.0, random_state=random_state
        )
        rng = np.random.RandomState(random_state + 1)
        X_noise = rng.randn(n_samples, n_features - n_informative) * 0.1
        X = np.hstack([X_info, X_noise])
    elif name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
        if n_features > 2:
            rng = np.random.RandomState(random_state)
            X = np.hstack([X, rng.randn(n_samples, n_features - 2) * 0.01])
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X.astype(np.float64), y


def time_sklearn(X, **params):
    """Time stock sklearn HDBSCAN."""
    # sklearn uses 'brute' not 'brute_force'
    if params.get("algorithm") == "brute_force":
        params["algorithm"] = "brute"
    est = sklearn_HDBSCAN(**params)
    start = time.perf_counter()
    est.fit(X)
    elapsed = time.perf_counter() - start
    return est.labels_, elapsed


def time_onedal(X, metric="euclidean", min_cluster_size=15, min_samples=10,
                algorithm="auto", metric_params=None):
    """Time oneDAL HDBSCAN (direct call, no dispatch overhead)."""
    est = onedal_HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        metric_params=metric_params,
    )
    start = time.perf_counter()
    est.fit(X)
    elapsed = time.perf_counter() - start
    return est.labels_, elapsed


def benchmark_run(X, y_true, dataset_name, n_samples, n_features,
                  metric="euclidean", min_cluster_size=15, min_samples=10,
                  algorithm="auto"):
    """Run one benchmark comparing sklearn vs oneDAL."""

    sklearn_params = dict(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
    )
    onedal_params = dict(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
    )
    if metric == "minkowski":
        sklearn_params["metric_params"] = {"p": 3}
        onedal_params["metric_params"] = {"p": 3}

    # Stock sklearn
    labels_sklearn, t_sklearn = time_sklearn(X, **sklearn_params)
    n_clusters_sklearn = len(set(labels_sklearn) - {-1})

    # oneDAL
    labels_onedal, t_onedal = time_onedal(X, **onedal_params)
    n_clusters_onedal = len(set(labels_onedal) - {-1})

    # Correctness: ARI between sklearn and oneDAL labels
    ari = adjusted_rand_score(labels_sklearn, labels_onedal)

    speedup = t_sklearn / t_onedal if t_onedal > 0 else float("inf")

    return {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "metric": metric,
        "algorithm": algorithm,
        "t_sklearn": t_sklearn,
        "t_onedal": t_onedal,
        "speedup": speedup,
        "ari_match": ari,
        "clusters_sklearn": n_clusters_sklearn,
        "clusters_onedal": n_clusters_onedal,
    }


# ── Benchmark configurations ─────────────────────────────────────────────────

def get_benchmark_configs():
    configs = []

    # 1. Scaling with dataset size (2D blobs)
    # Note: sklearn HDBSCAN is O(N^2), so >200k takes very long
    for n in [1000, 5000, 10000, 50000, 100000, 200000]:
        mcs = max(15, n // 500)
        configs.append({
            "dataset": "blobs", "n_samples": n, "n_features": 2,
            "metric": "euclidean", "min_cluster_size": mcs,
            "label": f"scale_{n}"
        })

    # 2. Scaling with dimensions (10k samples)
    for d in [2, 10, 50, 100]:
        configs.append({
            "dataset": "high_dim" if d > 2 else "blobs",
            "n_samples": 10000, "n_features": d,
            "metric": "euclidean", "min_cluster_size": 25,
            "label": f"dim_{d}"
        })

    # 3. Different cluster structures (10k samples)
    for ds in ["blobs", "varied_density", "noisy", "moons"]:
        configs.append({
            "dataset": ds, "n_samples": 10000, "n_features": 2,
            "metric": "euclidean", "min_cluster_size": 25,
            "label": f"struct_{ds}"
        })

    # 4. Different distance metrics (10k, 2D blobs)
    for m in ["euclidean", "manhattan", "chebyshev", "cosine", "minkowski"]:
        algo = "brute_force" if m == "cosine" else "auto"
        configs.append({
            "dataset": "blobs", "n_samples": 10000, "n_features": 2,
            "metric": m, "algorithm": algo, "min_cluster_size": 25,
            "label": f"metric_{m}"
        })

    # 5. Stress tests (capped to avoid sklearn O(N^2) timeout)
    configs.append({
        "dataset": "high_dim", "n_samples": 50000, "n_features": 50,
        "metric": "euclidean", "min_cluster_size": 50,
        "label": "stress_50k_50d"
    })
    configs.append({
        "dataset": "noisy", "n_samples": 50000, "n_features": 10,
        "metric": "euclidean", "min_cluster_size": 50,
        "label": "stress_50k_noisy"
    })

    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 110)
    print("  HDBSCAN Benchmark: stock scikit-learn vs oneDAL (via scikit-learn-intelex)")
    print("=" * 110)

    import sklearn
    print(f"  sklearn version: {sklearn.__version__}")
    try:
        import sklearnex
        print(f"  sklearnex version: {sklearnex.__version__}")
    except ImportError:
        pass
    print(f"  numpy version: {np.__version__}")
    print()

    configs = get_benchmark_configs()
    results = []

    categories = {
        "SCALING WITH DATASET SIZE (2D euclidean)": [c for c in configs if c["label"].startswith("scale_")],
        "SCALING WITH DIMENSIONS (N=10000, euclidean)": [c for c in configs if c["label"].startswith("dim_")],
        "CLUSTER STRUCTURES (N=10000, 2D, euclidean)": [c for c in configs if c["label"].startswith("struct_")],
        "DISTANCE METRICS (N=10000, 2D)": [c for c in configs if c["label"].startswith("metric_")],
        "STRESS TESTS": [c for c in configs if c["label"].startswith("stress_")],
    }

    for cat_name, cat_configs in categories.items():
        print(f"\n{'─' * 110}")
        print(f"  {cat_name}")
        print(f"{'─' * 110}")
        header = (f"{'Config':<20} {'N':>8} {'D':>4} {'Metric':<12} "
                  f"{'sklearn(s)':>11} {'oneDAL(s)':>11} {'Speedup':>9} "
                  f"{'ARI':>7} {'Cl_sk':>6} {'Cl_od':>6}")
        print(header)
        print("-" * len(header))

        for cfg in cat_configs:
            label = cfg["label"]
            n = cfg["n_samples"]
            d = cfg["n_features"]
            metric = cfg["metric"]
            algorithm = cfg.get("algorithm", "auto")
            mcs = cfg.get("min_cluster_size", 25)

            X, y = make_dataset(cfg["dataset"], n, d)

            try:
                result = benchmark_run(
                    X, y, cfg["dataset"], n, d,
                    metric=metric,
                    min_cluster_size=mcs,
                    min_samples=max(5, mcs // 2),
                    algorithm=algorithm,
                )
                result["label"] = label
                results.append(result)

                print(f"{label:<20} {n:>8} {d:>4} {metric:<12} "
                      f"{result['t_sklearn']:>11.3f} {result['t_onedal']:>11.3f} "
                      f"{result['speedup']:>8.1f}x "
                      f"{result['ari_match']:>7.4f} "
                      f"{result['clusters_sklearn']:>6} {result['clusters_onedal']:>6}")
            except Exception as e:
                print(f"{label:<20} {n:>8} {d:>4} {metric:<12} ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        print(f"\n{'=' * 110}")
        print("  SUMMARY")
        print(f"{'=' * 110}")

        speedups = [r["speedup"] for r in results]
        aris = [r["ari_match"] for r in results]
        total_sklearn = sum(r["t_sklearn"] for r in results)
        total_onedal = sum(r["t_onedal"] for r in results)

        print(f"  Total benchmarks run:  {len(results)}")
        print(f"  Median speedup:        {np.median(speedups):.1f}x")
        print(f"  Mean speedup:          {np.mean(speedups):.1f}x")
        print(f"  Min/Max speedup:       {min(speedups):.1f}x / {max(speedups):.1f}x")
        print(f"  Mean ARI (vs sklearn): {np.mean(aris):.4f}")
        print(f"  Min ARI (vs sklearn):  {min(aris):.4f}")
        print(f"  Total sklearn time:    {total_sklearn:.1f}s")
        print(f"  Total oneDAL time:     {total_onedal:.1f}s")
        print(f"  Overall speedup:       {total_sklearn / total_onedal:.1f}x")

        # Scaling analysis
        scale_results = [r for r in results if r["n_features"] == 2
                         and r["metric"] == "euclidean"
                         and r["dataset"] == "blobs"]
        if len(scale_results) > 1:
            print(f"\n  Speedup by dataset size (2D euclidean blobs):")
            for r in sorted(scale_results, key=lambda x: x["n_samples"]):
                bar = "#" * int(r["speedup"])
                print(f"    N={r['n_samples']:>8}: {r['speedup']:>7.1f}x  "
                      f"(sk={r['t_sklearn']:.3f}s, od={r['t_onedal']:.3f}s)  {bar}")

        # Cases where oneDAL is slower
        slow_cases = [r for r in results if r["speedup"] < 1.0]
        if slow_cases:
            print(f"\n  Cases where oneDAL is SLOWER than sklearn ({len(slow_cases)}):")
            for r in slow_cases:
                print(f"    {r['dataset']} N={r['n_samples']} D={r['n_features']} "
                      f"metric={r['metric']}: {r['speedup']:.2f}x")
        else:
            print(f"\n  oneDAL is faster in ALL {len(results)} test cases!")

        # Cases with ARI < 0.99
        mismatch_cases = [r for r in results if r["ari_match"] < 0.99]
        if mismatch_cases:
            print(f"\n  Cases with ARI < 0.99 vs sklearn ({len(mismatch_cases)}):")
            for r in mismatch_cases:
                print(f"    {r['dataset']} N={r['n_samples']} D={r['n_features']} "
                      f"metric={r['metric']}: ARI={r['ari_match']:.4f} "
                      f"(sk_cl={r['clusters_sklearn']}, od_cl={r['clusters_onedal']})")
        else:
            print(f"  Perfect correctness: ALL {len(results)} cases have ARI >= 0.99")

        # ── Recommendation ────────────────────────────────────────────────
        print(f"\n{'=' * 110}")
        print("  ANALYSIS & RECOMMENDATION")
        print(f"{'=' * 110}")

        fast_cases = [r for r in results if r["speedup"] > 2.0]
        very_fast = [r for r in results if r["speedup"] > 10.0]
        correct_cases = [r for r in results if r["ari_match"] > 0.95]

        print(f"  Cases with >2x speedup:   {len(fast_cases)}/{len(results)} ({100*len(fast_cases)/len(results):.0f}%)")
        print(f"  Cases with >10x speedup:  {len(very_fast)}/{len(results)} ({100*len(very_fast)/len(results):.0f}%)")
        print(f"  Cases with ARI > 0.95:    {len(correct_cases)}/{len(results)} ({100*len(correct_cases)/len(results):.0f}%)")

        print(f"\n  TOP 5 BEST USE CASES (highest speedup):")
        for r in sorted(results, key=lambda x: x["speedup"], reverse=True)[:5]:
            print(f"    {r['dataset']} N={r['n_samples']:>7} D={r['n_features']:>3} "
                  f"{r['metric']:<12}: {r['speedup']:>7.1f}x  ARI={r['ari_match']:.4f}")

        print(f"\n  BOTTOM 3 WEAKEST CASES (lowest speedup):")
        for r in sorted(results, key=lambda x: x["speedup"])[:3]:
            print(f"    {r['dataset']} N={r['n_samples']:>7} D={r['n_features']:>3} "
                  f"{r['metric']:<12}: {r['speedup']:>7.1f}x  ARI={r['ari_match']:.4f}")

        print(f"\n  WHERE IT MAKES SENSE TO USE oneDAL HDBSCAN:")
        if np.median(speedups) > 2.0:
            print(f"    - Medium to large datasets (N > 5000): consistent speedup")
        if any(r["speedup"] > 10 for r in results if r["n_samples"] >= 50000):
            print(f"    - Large-scale clustering (N > 50k): dramatic speedup (>10x)")
        if any(r["speedup"] > 1.5 for r in results if r["n_features"] >= 50):
            print(f"    - High-dimensional data: maintains speedup advantage")
        metric_speedups = {r["metric"]: r["speedup"] for r in results if r["label"].startswith("metric_")}
        fast_metrics = [m for m, s in metric_speedups.items() if s > 2.0]
        if fast_metrics:
            print(f"    - All distance metrics benefit: {', '.join(fast_metrics)}")
        print(f"    - Production pipelines where HDBSCAN is a bottleneck")

        print(f"\n  WHERE STOCK SKLEARN MAY SUFFICE:")
        if any(r["speedup"] < 2.0 for r in results if r["n_samples"] <= 1000):
            print(f"    - Small datasets (N < 1000): overhead may reduce advantage")
        print(f"    - One-off exploratory analysis where setup cost matters")
        print(f"    - When allow_single_cluster=True or leaf selection is needed")

        if np.median(speedups) > 2.0 and np.mean(aris) > 0.95:
            print(f"\n  VERDICT: STRONG YES")
            print(f"  oneDAL HDBSCAN provides {np.median(speedups):.0f}x median speedup with {np.mean(aris):.4f} mean ARI.")
            print(f"  It is worth extending scikit-learn-intelex with HDBSCAN support.")
        elif np.median(speedups) > 1.0 and np.mean(aris) > 0.90:
            print(f"\n  VERDICT: YES, with caveats")
        else:
            print(f"\n  VERDICT: NEEDS MORE WORK")


if __name__ == "__main__":
    main()
