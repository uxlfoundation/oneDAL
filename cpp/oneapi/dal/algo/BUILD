package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_collect_modules",
    "dal_collect_parameters",
    "dal_collect_test_suites",
)

PARAMETRIZED_ALGOS = [
    "covariance",
    "decision_forest",
    "linear_regression",
]

ALGOS = [
    "basic_statistics",
    "chebyshev_distance",
    "connected_components",
    "cosine_distance",
    "dbscan",
    "decision_tree",
    "finiteness_checker",
    "jaccard",
    "kmeans",
    "kmeans_init",
    "knn",
    "linear_kernel",
    "logistic_regression",
    "logloss_objective",
    "louvain",
    "minkowski_distance",
    "objective_function",
    "pca",
    "polynomial_kernel",
    "rbf_kernel",
    "sigmoid_kernel",
    "shortest_paths",
    "spectral_embedding",
    "subgraph_isomorphism",
    "svm",
    "triangle_counting",
] + PARAMETRIZED_ALGOS

dal_collect_modules(
    name = "algo",
    modules = ALGOS,
    root = "@onedal//cpp/oneapi/dal/algo",
)

dal_collect_parameters(
    name = "parameters",
    modules = PARAMETRIZED_ALGOS,
    root = "@onedal//cpp/oneapi/dal/algo",
)

dal_collect_test_suites(
    name = "tests",
    root = "@onedal//cpp/oneapi/dal/algo",
    modules = ALGOS,
)
