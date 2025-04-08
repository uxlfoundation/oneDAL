load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_example_suite",
    "dal_algo_example_suite",
)

dal_module(
    name = "example_util",
    hdrs = glob(["source/example_util/*.hpp"]),
    includes = [ "source" ],
)

_TEST_DEPS = [
    ":example_util",
]

_DATA_DEPS = [
    "@onedal//examples/oneapi:data",
]

dal_example_suite(
    name = "graph",
    compile_as = [ "c++" ],
    srcs = glob(["source/graph/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal/graph",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)

dal_example_suite(
    name = "table",
    compile_as = [ "c++" ],
    srcs = glob(["source/table/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal/table",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)

dal_example_suite(
    name = "kmeans_init",
    compile_as = [ "c++" ],
    srcs = glob(["source/kmeans_init/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo:kmeans",
        "@onedal//cpp/oneapi/dal/algo:kmeans_init",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)

dal_example_suite(
    name = "misc",
    compile_as = [ "c++" ],
    srcs = glob(["source/misc/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal:io",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)

dal_example_suite(
    name = "jaccard",
    compile_as = [ "c++" ],
    srcs = glob(["source/jaccard/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo:jaccard",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_algo_example_suite(
    algos = [
        "basic_statistics",
        "connected_components",
        "covariance",
        "dbscan",
        "decision_forest",
        "kmeans",
        "knn",
        "linear_kernel",
        "linear_regression",
        "louvain",
        "objective_function",
        "pca",
        "polynomial_kernel",
        "rbf_kernel",
        "shortest_paths",
        "sigmoid_kernel",
        "subgraph_isomorphism",
        "svm",
        "triangle_counting",
    ],
    compile_as = [ "c++" ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)
