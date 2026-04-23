"""
Comprehensive HDBSCAN comparison: oneDAL (CPU) vs scikit-learn.

Tests every (metric, method) combination on multiple datasets.
Generates data, saves as CSV, runs both oneDAL examples and sklearn,
compares labels using ARI.
"""

import numpy as np
import os
import subprocess
import sys
import tempfile
import warnings
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore", category=FutureWarning)

ONEDAL_DIR = "/localdisk2/mkl/asolovev/oneDAL"
EXAMPLE_DIR = f"{ONEDAL_DIR}/__release_lnx/daal/latest/examples/oneapi/dpc/_cmake_results/intel_intel64_so"
BF_EXE = f"{EXAMPLE_DIR}/hdbscan_brute_force_batch"
KD_EXE = f"{EXAMPLE_DIR}/hdbscan_kd_tree_batch"


def generate_two_clusters(n=50, dim=2, sep=10.0, noise=0.3, seed=42):
    rng = np.random.RandomState(seed)
    c0 = rng.randn(n, dim) * noise
    c1 = rng.randn(n, dim) * noise + sep
    return np.vstack([c0, c1]).astype(np.float32)


def generate_three_1d(n=20, sep=10.0, noise=0.2, seed=42):
    rng = np.random.RandomState(seed)
    c0 = rng.randn(n, 1) * noise
    c1 = rng.randn(n, 1) * noise + sep
    c2 = rng.randn(n, 1) * noise + 2 * sep
    return np.vstack([c0, c1, c2]).astype(np.float32)


def generate_cosine(n=50, seed=42):
    rng = np.random.RandomState(seed)
    # Cluster 0: direction ~(1, 0)
    a0 = rng.randn(n) * 0.05
    r0 = rng.uniform(5, 15, n)
    c0 = np.column_stack([r0 * np.cos(a0), r0 * np.sin(a0)])
    # Cluster 1: direction ~(0, 1)
    a1 = np.pi / 2 + rng.randn(n) * 0.05
    r1 = rng.uniform(5, 15, n)
    c1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    return np.vstack([c0, c1]).astype(np.float32)


def generate_high_dim(n=30, dim=10, sep=10.0, noise=0.3, seed=42):
    return generate_two_clusters(n, dim, sep, noise, seed)


def save_csv(data, path):
    np.savetxt(path, data, delimiter=',', fmt='%.8f')


def run_sklearn(data, metric, mcs, ms, degree=2.0):
    kwargs = dict(
        min_cluster_size=mcs,
        min_samples=ms,
        metric=metric,
        cluster_selection_method='eom',
    )
    if metric == 'minkowski':
        kwargs['metric_params'] = {'p': degree}
    model = SklearnHDBSCAN(**kwargs)
    return model.fit_predict(data)


def run_onedal(data_path, method, metric, mcs, ms, degree=2.0):
    exe = BF_EXE if method == 'brute_force' else KD_EXE
    env = os.environ.copy()
    env['HDBSCAN_DATA_PATH'] = data_path
    env['HDBSCAN_MIN_CLUSTER_SIZE'] = str(mcs)
    env['HDBSCAN_MIN_SAMPLES'] = str(ms)
    env['HDBSCAN_METRIC'] = metric
    env['HDBSCAN_DEGREE'] = str(degree)

    try:
        result = subprocess.run(
            [exe], capture_output=True, text=True, env=env, timeout=30
        )
        output = result.stdout
        # Parse "Labels: 0 0 1 1 ..." line
        for line in output.split('\n'):
            if line.startswith('Labels:'):
                parts = line.replace('Labels:', '').strip().split()
                return np.array([int(x) for x in parts])
        # If we see it on OpenCL device, there may be multiple runs - take first
        labels_found = []
        for line in output.split('\n'):
            if 'Labels:' in line:
                parts = line.split('Labels:')[1].strip().split()
                labels_found.append(np.array([int(x) for x in parts]))
        if labels_found:
            return labels_found[0]
        print(f"    WARNING: Could not parse labels from output")
        print(f"    stdout: {output[:500]}")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def check_same_partition(a, b):
    """Check if two label arrays define the same partition (ARI)."""
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    return adjusted_rand_score(a, b)


# ============================================================================
# Test matrix
# ============================================================================

DATASETS = {
    'two_clusters_2d': {
        'generator': lambda: generate_two_clusters(50, 2, 10.0, 0.3, seed=42),
        'mcs': 10, 'ms': 5,
    },
    'three_clusters_1d': {
        'generator': lambda: generate_three_1d(20, 10.0, 0.2, seed=42),
        'mcs': 5, 'ms': 5,
    },
    'high_dim_10d': {
        'generator': lambda: generate_high_dim(30, 10, 10.0, 0.3, seed=42),
        'mcs': 10, 'ms': 5,
    },
}

COSINE_DATASET = {
    'cosine_angular_2d': {
        'generator': lambda: generate_cosine(50, seed=42),
        'mcs': 10, 'ms': 5,
    },
}

# Metrics and their supported methods
METRICS = {
    'euclidean':  {'degree': 2.0, 'methods': ['brute_force', 'kd_tree']},
    'manhattan':  {'degree': 2.0, 'methods': ['brute_force', 'kd_tree']},
    'chebyshev':  {'degree': 2.0, 'methods': ['brute_force', 'kd_tree']},
    'minkowski':  {'degree': 3.0, 'methods': ['brute_force', 'kd_tree']},
    'cosine':     {'degree': 2.0, 'methods': ['brute_force']},
}

# Additional Minkowski equivalence checks
EQUIV_CHECKS = [
    ('minkowski', 1.0, 'manhattan', 2.0),  # Minkowski(p=1) == Manhattan
    ('minkowski', 2.0, 'euclidean', 2.0),  # Minkowski(p=2) == Euclidean
]


def main():
    print("=" * 80)
    print("HDBSCAN Metric Validation: oneDAL (CPU) vs scikit-learn")
    print("=" * 80)

    all_pass = True
    mismatches = []

    # Generate all datasets and save as CSV
    csv_files = {}
    for ds_name, ds_info in {**DATASETS, **COSINE_DATASET}.items():
        data = ds_info['generator']()
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        save_csv(data, f.name)
        f.close()
        csv_files[ds_name] = (f.name, data, ds_info['mcs'], ds_info['ms'])

    try:
        # Test each (metric, method, dataset) combination
        for metric_name, metric_info in METRICS.items():
            degree = metric_info['degree']
            methods = metric_info['methods']

            # Choose datasets
            if metric_name == 'cosine':
                ds_iter = COSINE_DATASET.items()
            else:
                ds_iter = DATASETS.items()

            for ds_name, ds_info in ds_iter:
                csv_path, data, mcs, ms = csv_files[ds_name]

                # Run sklearn
                sk_labels = run_sklearn(data, metric_name, mcs, ms, degree)
                sk_clusters = len(set(sk_labels) - {-1})
                sk_noise = np.sum(sk_labels == -1)

                for method in methods:
                    # Run oneDAL
                    dal_labels = run_onedal(csv_path, method, metric_name, mcs, ms, degree)

                    if dal_labels is None:
                        status = "FAIL (no output)"
                        all_pass = False
                        mismatches.append(f"{ds_name}/{metric_name}/{method}: no output")
                    else:
                        dal_clusters = len(set(dal_labels) - {-1})
                        dal_noise = np.sum(dal_labels == -1)
                        ari = check_same_partition(sk_labels, dal_labels)

                        if ari is not None and ari >= 0.99:
                            status = f"PASS (ARI={ari:.4f}, clusters={dal_clusters}, noise={dal_noise})"
                        else:
                            status = f"MISMATCH (ARI={ari}, sk_clusters={sk_clusters}/{sk_noise}n, dal_clusters={dal_clusters}/{dal_noise}n)"
                            all_pass = False
                            mismatches.append(
                                f"{ds_name}/{metric_name}/{method}: ARI={ari}, "
                                f"sklearn={sk_clusters}c/{sk_noise}n, oneDAL={dal_clusters}c/{dal_noise}n"
                            )

                    deg_str = f"(p={degree})" if metric_name == 'minkowski' else ""
                    print(f"  {ds_name:20s} | {metric_name:10s}{deg_str:6s} | {method:12s} | {status}")

        # Equivalence checks
        print("\n" + "=" * 80)
        print("Minkowski equivalence checks")
        print("=" * 80)

        for ds_name, ds_info in DATASETS.items():
            csv_path, data, mcs, ms = csv_files[ds_name]

            for metric_a, degree_a, metric_b, degree_b in EQUIV_CHECKS:
                for method in ['brute_force', 'kd_tree']:
                    labels_a = run_onedal(csv_path, method, metric_a, mcs, ms, degree_a)
                    labels_b = run_onedal(csv_path, method, metric_b, mcs, ms, degree_b)

                    if labels_a is None or labels_b is None:
                        status = "FAIL (no output)"
                        all_pass = False
                    else:
                        ari = check_same_partition(labels_a, labels_b)
                        n_cl_a = len(set(labels_a) - {-1})
                        n_cl_b = len(set(labels_b) - {-1})
                        if ari is not None and ari >= 0.99:
                            status = f"PASS (ARI={ari:.4f})"
                        else:
                            status = f"MISMATCH (ARI={ari}, {metric_a}(p={degree_a})={n_cl_a}c, {metric_b}={n_cl_b}c)"
                            all_pass = False
                            mismatches.append(
                                f"{ds_name}/{metric_a}(p={degree_a}) vs {metric_b}/{method}: ARI={ari}"
                            )

                    print(f"  {ds_name:20s} | {metric_a}(p={degree_a}) == {metric_b:10s} | {method:12s} | {status}")

        # Cross-method consistency (brute_force vs kd_tree same metric)
        print("\n" + "=" * 80)
        print("Cross-method consistency: brute_force vs kd_tree")
        print("=" * 80)

        for ds_name, ds_info in DATASETS.items():
            csv_path, data, mcs, ms = csv_files[ds_name]

            for metric_name in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
                degree = 3.0 if metric_name == 'minkowski' else 2.0

                bf_labels = run_onedal(csv_path, 'brute_force', metric_name, mcs, ms, degree)
                kd_labels = run_onedal(csv_path, 'kd_tree', metric_name, mcs, ms, degree)

                if bf_labels is None or kd_labels is None:
                    status = "FAIL (no output)"
                    all_pass = False
                else:
                    ari = check_same_partition(bf_labels, kd_labels)
                    if ari is not None and ari >= 0.99:
                        status = f"PASS (ARI={ari:.4f})"
                    else:
                        status = f"MISMATCH (ARI={ari})"
                        all_pass = False
                        mismatches.append(f"{ds_name}/{metric_name}/bf_vs_kd: ARI={ari}")

                deg_str = f"(p={degree})" if metric_name == 'minkowski' else ""
                print(f"  {ds_name:20s} | {metric_name:10s}{deg_str:6s} | bf vs kd | {status}")

    finally:
        # Cleanup temp files
        for name, (path, _, _, _) in csv_files.items():
            os.unlink(path)

    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES ({len(mismatches)}):")
        for m in mismatches:
            print(f"  - {m}")
    print("=" * 80)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
