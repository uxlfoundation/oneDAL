#!/usr/bin/env python3
"""Validate HDBSCAN allow_single_cluster and cluster_selection_method
against stock sklearn across multiple datasets and parameter combinations."""

import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import HDBSCAN as sklearn_HDBSCAN

# Direct import from onedal layer (avoids patching complexity)
from onedal.cluster import HDBSCAN as onedal_HDBSCAN


def make_test_datasets():
    """Create datasets that exercise different clustering scenarios."""
    datasets = {}

    # Well-separated clusters (easy)
    X, y = make_blobs(n_samples=500, centers=5, cluster_std=0.5, random_state=42)
    datasets["well_separated_5c"] = (X, y)

    # Overlapping clusters (harder, tests allow_single_cluster)
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)
    datasets["overlapping_3c"] = (X, y)

    # Two well-separated clusters
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=0.3, random_state=42)
    datasets["two_clusters"] = (X, y)

    # Many small clusters (tests leaf selection)
    X, y = make_blobs(n_samples=600, centers=8, cluster_std=0.8, random_state=42)
    datasets["many_small_8c"] = (X, y)

    # High-dimensional
    X, y = make_blobs(n_samples=400, centers=4, n_features=10, cluster_std=1.0, random_state=42)
    datasets["high_dim_4c"] = (X, y)

    return datasets


def run_sklearn(X, min_cluster_size, cluster_selection_method, allow_single_cluster):
    clf = sklearn_HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    t0 = time.time()
    clf.fit(X)
    t1 = time.time()
    return clf.labels_, t1 - t0


def run_onedal(X, min_cluster_size, cluster_selection_method, allow_single_cluster):
    clf = onedal_HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    t0 = time.time()
    clf.fit(X)
    t1 = time.time()
    return clf.labels_, t1 - t0


def count_clusters(labels):
    return len(set(labels) - {-1})


def count_noise(labels):
    return np.sum(labels == -1)


def main():
    datasets = make_test_datasets()

    param_combos = [
        {"cluster_selection_method": "eom", "allow_single_cluster": False},
        {"cluster_selection_method": "eom", "allow_single_cluster": True},
        {"cluster_selection_method": "leaf", "allow_single_cluster": False},
        {"cluster_selection_method": "leaf", "allow_single_cluster": True},
    ]

    min_cluster_sizes = [5, 15]

    print("=" * 90)
    print("HDBSCAN Parameter Validation: oneDAL vs sklearn")
    print("=" * 90)

    results = []
    total = 0
    passed = 0
    failed = 0

    for ds_name, (X, y_true) in datasets.items():
        print(f"\n{'─' * 90}")
        print(f"Dataset: {ds_name} ({X.shape[0]} samples, {X.shape[1]} features)")
        print(f"{'─' * 90}")

        for mcs in min_cluster_sizes:
            for params in param_combos:
                csm = params["cluster_selection_method"]
                asc = params["allow_single_cluster"]
                label = f"mcs={mcs}, sel={csm}, asc={asc}"

                total += 1

                try:
                    sk_labels, sk_time = run_sklearn(X, mcs, csm, asc)
                    od_labels, od_time = run_onedal(X, mcs, csm, asc)

                    ari = adjusted_rand_score(sk_labels, od_labels)
                    sk_nc = count_clusters(sk_labels)
                    od_nc = count_clusters(od_labels)
                    sk_noise = count_noise(sk_labels)
                    od_noise = count_noise(od_labels)
                    speedup = sk_time / od_time if od_time > 0 else float("inf")

                    status = "PASS" if ari > 0.95 else "FAIL"
                    if status == "PASS":
                        passed += 1
                    else:
                        failed += 1

                    print(
                        f"  {status} | {label:40s} | "
                        f"ARI={ari:.4f} | clusters: sk={sk_nc} od={od_nc} | "
                        f"noise: sk={sk_noise} od={od_noise} | "
                        f"{speedup:.1f}x"
                    )

                    if status == "FAIL":
                        print(f"         *** MISMATCH: ARI={ari:.4f}, expected > 0.95")

                    results.append({
                        "dataset": ds_name,
                        "label": label,
                        "ari": ari,
                        "sk_clusters": sk_nc,
                        "od_clusters": od_nc,
                        "status": status,
                    })

                except Exception as e:
                    failed += 1
                    print(f"  ERROR | {label:40s} | {e}")
                    results.append({
                        "dataset": ds_name,
                        "label": label,
                        "ari": -1,
                        "status": "ERROR",
                        "error": str(e),
                    })

    print(f"\n{'=' * 90}")
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 90}")

    # Print failures
    failures = [r for r in results if r["status"] != "PASS"]
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f['dataset']}: {f['label']} (ARI={f.get('ari', 'N/A')})")
    else:
        print("\nAll tests PASSED!")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
