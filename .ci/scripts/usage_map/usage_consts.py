#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

"""Shared constants and helpers of the usage-map scripts.

USAGE_MAP_WORK (default: cwd) is the work directory; every script reads and writes under it.
Consumer groups (the `consumer` field of units-<build>.json):
  REAL      runs the product: sklearnex (CPU, GPU), scikit-learn_bench workloads, daal4py estimators,
            model builders
  PUBLIC    daal4py public-API tests
  FILTERED  oneDAL's own examples and tests (C++ examples, Bazel tests, onedal python package tests);
            they show that code is exercised, not that users need it
"""
import json
import os
import re

U = os.environ.get("USAGE_MAP_WORK", os.getcwd())
REAL = ["sklearnex", "sklearnex-gpu", "workloads", "daal4py", "model-builders"]
PUBLIC = ["daal4py-api"]
FILTERED = ["examples-oneapi", "examples-daal", "onedal-tests", "onedal-py-tests"]
CPU = re.compile(r"\((?:daal::)?(?:internal::)?CpuType\)\d+|\b(?:sse2|sse42|avx2|avx512)\b")

# static surfaces (2026.1.0): oneapi algos bound in sklearnex onedal/*.cpp,
# daal dirs reachable from daal4py public names, daal dirs sklearnex python calls
PYBIND = set("basic_statistics chebyshev_distance cosine_distance covariance dbscan decision_forest "
             "finiteness_checker kmeans kmeans_init knn linear_kernel linear_regression logistic_regression "
             "minkowski_distance newton_cg pca polynomial_kernel rbf_kernel sigmoid_kernel svm".split())
D4P = set("cholesky cordistance cosdistance covariance dbscan dtrees elastic_net em engines implicit_als "
          "k_nearest_neighbors kernel_function kmeans lasso_regression linear_model linear_regression "
          "logistic_regression low_order_moments multiclassclassifier normalization objective_function "
          "optimization_solver pca pivoted_qr qr regression ridge_regression svd svm tsne classifier".split())
SKX_D4P = set("kmeans optimization_solver objective_function linear_regression engines dtrees tsne svm "
              "ridge_regression pca multiclassclassifier logistic_regression lasso_regression kernel_function "
              "k_nearest_neighbors elastic_net dbscan covariance cosdistance cordistance linear_model "
              "regression classifier".split())


def static_tag(api, comp):
    if api == "oneapi":
        return "sklearnex-pybind" if comp in PYBIND else ""
    if comp in SKX_D4P:
        return "sklearnex-via-d4p"
    return "d4p-public" if comp in D4P else ""


def area(path):
    """Map a source path to (api, component, sub)."""
    p = path.split("/")
    if p[1] == "daal":
        if p[2] == "src" and p[3] == "algorithms" and len(p) > 5:
            return "daal", p[4], p[5] if len(p) > 6 else "-"
        if p[2] == "src" and p[3] == "algorithms":
            return "daal", "shared:algorithms-base", "-"
        if p[2] == "include" and p[3] == "algorithms" and len(p) > 5:
            return "daal", p[4], "include"
        return "daal", "shared:" + "/".join(p[2:4]), "-"
    if p[1] == "oneapi":
        if p[3] == "algo" and len(p) > 5:
            return "oneapi", p[4], p[5] if len(p) > 6 else "-"
        return "oneapi", "shared:" + "/".join(p[3:4] if len(p) > 4 else ["root"]), "-"
    return p[1], "shared:other", "-"


def load_units(build):
    """units-<build>.json plus scikit-learn_bench cases (units-bench.json, written by run_bench.py)."""
    us = json.load(open(f"{U}/units-{build}.json"))
    bench = f"{U}/units-bench.json"
    if os.path.exists(bench):
        us += json.load(open(bench))
    return us
