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
    name = "table",
    compile_as = [ "dpc++" ],
    srcs = glob(["source/table/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal/table",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)

dal_example_suite(
    name = "kmeans_init",
    compile_as = [ "dpc++" ],
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
    compile_as = [ "dpc++" ],
    srcs = glob(["source/misc/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal:io",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)

dal_algo_example_suite(
    algos = [
        "basic_statistics",
        "correlation_distance",
        "covariance",
        "decision_forest",
        "kmeans",
        "dbscan",
        "knn",
        "linear_kernel",
        "linear_regression",
        "logistic_regression",
        "pca",
        "rbf_kernel",
        "svm",
    ],
    compile_as = [ "dpc++" ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)
