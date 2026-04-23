"""
Stress test: HDBSCAN oneDAL vs sklearn on harder datasets where
clusters overlap and noise is present. These are the cases where
floating-point differences in distance computation are most likely
to cause mismatches.
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
        result = subprocess.run([exe], capture_output=True, text=True, env=env, timeout=30)
        for line in result.stdout.split('\n'):
            if line.startswith('Labels:'):
                parts = line.replace('Labels:', '').strip().split()
                return np.array([int(x) for x in parts])
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    print("=" * 80)
    print("HDBSCAN Stress Test: harder datasets (overlapping clusters, noise)")
    print("=" * 80)

    all_pass = True
    mismatches = []

    rng = np.random.RandomState(123)

    # ============================
    # Dataset 1: overlapping blobs
    # ============================
    # Two clusters with moderate overlap (sep=3.0 vs noise=1.0)
    c0 = rng.randn(80, 3).astype(np.float32) * 1.0
    c1 = (rng.randn(80, 3).astype(np.float32) * 1.0) + 3.0
    data_overlap = np.vstack([c0, c1])

    # ============================
    # Dataset 2: varying density
    # ============================
    # Tight cluster + loose cluster
    c_tight = rng.randn(60, 2).astype(np.float32) * 0.1
    c_loose = rng.randn(60, 2).astype(np.float32) * 2.0 + 15.0
    noise_pts = rng.uniform(-5, 20, (20, 2)).astype(np.float32)
    data_density = np.vstack([c_tight, c_loose, noise_pts])

    # ============================
    # Dataset 3: many small clusters
    # ============================
    clusters = []
    for i in range(5):
        center = rng.uniform(-20, 20, 2).astype(np.float32)
        cl = rng.randn(15, 2).astype(np.float32) * 0.3 + center
        clusters.append(cl)
    noise_pts2 = rng.uniform(-25, 25, (10, 2)).astype(np.float32)
    data_many = np.vstack(clusters + [noise_pts2])

    # ============================
    # Dataset 4: cosine with overlap
    # ============================
    # Wider angular noise so clusters partially overlap
    a0 = rng.randn(60) * 0.15  # wider angular noise
    r0 = rng.uniform(3, 20, 60)
    c_cos0 = np.column_stack([r0 * np.cos(a0), r0 * np.sin(a0)])
    a1 = np.pi / 2 + rng.randn(60) * 0.15
    r1 = rng.uniform(3, 20, 60)
    c_cos1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    data_cosine_hard = np.vstack([c_cos0, c_cos1]).astype(np.float32)

    # ============================
    # Dataset 5: high-dimensional with noise dims
    # ============================
    # 2 real clusters in first 3 dims, rest are noise dimensions
    real0 = rng.randn(50, 3).astype(np.float32) * 0.5
    real1 = rng.randn(50, 3).astype(np.float32) * 0.5 + 5.0
    noise_dims = rng.randn(100, 7).astype(np.float32) * 0.5
    data_noisy_dims = np.hstack([np.vstack([real0, real1]), noise_dims])

    datasets = {
        'overlapping_3d':   (data_overlap, 15, 5),
        'varying_density':  (data_density, 10, 5),
        'many_small':       (data_many, 5, 5),
        'noisy_dims_10d':   (data_noisy_dims, 15, 5),
    }

    cosine_datasets = {
        'cosine_overlap':   (data_cosine_hard, 10, 5),
    }

    metrics_methods = {
        'euclidean':  (['brute_force', 'kd_tree'], 2.0),
        'manhattan':  (['brute_force', 'kd_tree'], 2.0),
        'chebyshev':  (['brute_force', 'kd_tree'], 2.0),
        'minkowski':  (['brute_force', 'kd_tree'], 3.0),
        'cosine':     (['brute_force'], 2.0),
    }

    csv_files = {}
    try:
        # Save all datasets
        for name, (data, mcs, ms) in {**datasets, **cosine_datasets}.items():
            f = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            save_csv(data, f.name)
            f.close()
            csv_files[name] = f.name

        for metric_name, (methods, degree) in metrics_methods.items():
            ds_iter = cosine_datasets if metric_name == 'cosine' else datasets

            for ds_name, (data, mcs, ms) in ds_iter.items():
                csv_path = csv_files[ds_name]
                sk_labels = run_sklearn(data, metric_name, mcs, ms, degree)
                sk_clusters = len(set(sk_labels) - {-1})
                sk_noise = int(np.sum(sk_labels == -1))

                for method in methods:
                    dal_labels = run_onedal(csv_path, method, metric_name, mcs, ms, degree)

                    if dal_labels is None:
                        status = "FAIL (no output)"
                        all_pass = False
                        mismatches.append(f"{ds_name}/{metric_name}/{method}: no output")
                    else:
                        dal_clusters = len(set(dal_labels) - {-1})
                        dal_noise = int(np.sum(dal_labels == -1))
                        ari = adjusted_rand_score(sk_labels, dal_labels)

                        if ari >= 0.95:
                            status = f"PASS (ARI={ari:.4f}, sk={sk_clusters}c/{sk_noise}n, dal={dal_clusters}c/{dal_noise}n)"
                        elif ari >= 0.80:
                            status = f"WARN (ARI={ari:.4f}, sk={sk_clusters}c/{sk_noise}n, dal={dal_clusters}c/{dal_noise}n)"
                            # Warn but don't fail for near-boundary cases
                        else:
                            status = f"MISMATCH (ARI={ari:.4f}, sk={sk_clusters}c/{sk_noise}n, dal={dal_clusters}c/{dal_noise}n)"
                            all_pass = False
                            mismatches.append(
                                f"{ds_name}/{metric_name}/{method}: ARI={ari:.4f}, "
                                f"sk={sk_clusters}c/{sk_noise}n, dal={dal_clusters}c/{dal_noise}n"
                            )

                    deg = f"(p={degree})" if metric_name == 'minkowski' else ""
                    print(f"  {ds_name:20s} | {metric_name:10s}{deg:6s} | {method:12s} | {status}")

        # brute_force vs kd_tree on stress datasets
        print("\n  --- Cross-method consistency on stress datasets ---")
        for ds_name, (data, mcs, ms) in datasets.items():
            csv_path = csv_files[ds_name]
            for metric_name in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
                degree = 3.0 if metric_name == 'minkowski' else 2.0
                bf = run_onedal(csv_path, 'brute_force', metric_name, mcs, ms, degree)
                kd = run_onedal(csv_path, 'kd_tree', metric_name, mcs, ms, degree)
                if bf is not None and kd is not None:
                    ari = adjusted_rand_score(bf, kd)
                    bf_cl = len(set(bf) - {-1})
                    kd_cl = len(set(kd) - {-1})
                    if ari >= 0.95:
                        status = f"PASS (ARI={ari:.4f})"
                    else:
                        status = f"MISMATCH (ARI={ari:.4f}, bf={bf_cl}c, kd={kd_cl}c)"
                        all_pass = False
                        mismatches.append(f"{ds_name}/{metric_name}/bf_vs_kd: ARI={ari:.4f}")
                else:
                    status = "FAIL (no output)"
                    all_pass = False
                deg = f"(p={degree})" if metric_name == 'minkowski' else ""
                print(f"  {ds_name:20s} | {metric_name:10s}{deg:6s} | bf vs kd     | {status}")

    finally:
        for path in csv_files.values():
            os.unlink(path)

    print("\n" + "=" * 80)
    if all_pass:
        print("ALL STRESS TESTS PASSED")
    else:
        print(f"FAILURES ({len(mismatches)}):")
        for m in mismatches:
            print(f"  - {m}")
    print("=" * 80)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
