#!/usr/bin/env python3
"""Verify HDBSCAN works through the sklearnex patching mechanism."""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.datasets import make_blobs

# Test 1: Direct onedal import works
print("Test 1: Direct onedal import...")
from onedal.cluster import HDBSCAN as onedal_HDBSCAN
X, y = make_blobs(n_samples=200, centers=3, random_state=42)
clf = onedal_HDBSCAN(min_cluster_size=10, cluster_selection_method="leaf", allow_single_cluster=True)
clf.fit(X)
n_clusters = len(set(clf.labels_) - {-1})
print(f"  onedal direct: {n_clusters} clusters (leaf, allow_single=True)")
assert n_clusters > 0, "Expected at least 1 cluster"
print("  PASS")

# Test 2: Patching mechanism
print("\nTest 2: Patching mechanism...")
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import HDBSCAN
clf = HDBSCAN(min_cluster_size=10, cluster_selection_method="eom")
clf.fit(X)
n_clusters = len(set(clf.labels_) - {-1})
print(f"  patched sklearn: {n_clusters} clusters (eom)")
# Check it's actually using onedal
print(f"  class: {type(clf).__module__}.{type(clf).__name__}")
assert "sklearnex" in type(clf).__module__, f"Expected sklearnex class, got {type(clf).__module__}"
print("  PASS")

# Test 3: Leaf selection through patching
print("\nTest 3: Leaf selection through patching...")
clf = HDBSCAN(min_cluster_size=10, cluster_selection_method="leaf")
clf.fit(X)
n_clusters_leaf = len(set(clf.labels_) - {-1})
print(f"  patched sklearn: {n_clusters_leaf} clusters (leaf)")
print("  PASS")

# Test 4: allow_single_cluster through patching
print("\nTest 4: allow_single_cluster through patching...")
clf = HDBSCAN(min_cluster_size=10, allow_single_cluster=True)
clf.fit(X)
n_clusters_asc = len(set(clf.labels_) - {-1})
print(f"  patched sklearn: {n_clusters_asc} clusters (allow_single_cluster=True)")
print("  PASS")

print("\n" + "=" * 50)
print("All patching tests PASSED")
print("=" * 50)
