load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_example_suite",
    "dal_algo_example_suite",
)

dal_module(
    name = "example_util",
    hdrs = glob(["source/utils/*.h"]),
    includes = [ "source/utils" ],
)

_TEST_DEPS = [
    ":example_util", "@opencl//:opencl_binary",
]

_DATA_DEPS = [
    "@onedal//examples/daal:data",
]

# dal_example_suite(
#     name = "graph",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/graph/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal/graph",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS,
# )

# dal_example_suite(
#     name = "table",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/table/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal/table",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS,
# )

# dal_example_suite(
#     name = "kmeans_init",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/kmeans_init/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal/algo:kmeans",
#         "@onedal//cpp/oneapi/dal/algo:kmeans_init",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS,
# )

# dal_example_suite(
#     name = "misc",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/misc/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/oneapi/dal:io",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS,
# )

#"@onedal//cpp/daal/src/algorithms/dtrees/forest:kernel",
dal_example_suite(
    name = "association_rules",
    compile_as = [ "c++" ],
    srcs = glob(["source/association_rules/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/assocrules:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "boosting",
    compile_as = [ "c++" ],
    srcs = glob(["source/boosting/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/adaboost:kernel",
        "@onedal//cpp/daal/src/algorithms/brownboost:kernel",
        "@onedal//cpp/daal/src/algorithms/logitboost:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "cholesky",
    compile_as = [ "c++" ],
    srcs = glob(["source/cholesky/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/cholesky:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

# dal_example_suite(
#     name = "decision_tree",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/decision_tree/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/daal/src/algorithms/decision_tree:kernel",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS + [
#         "@tbb//:tbb",
#     ],
# )

dal_example_suite(
    name = "distance",
    compile_as = [ "c++" ],
    srcs = glob(["source/distance/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/cordistance:kernel",
         "@onedal//cpp/daal/src/algorithms/cosdistance:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "distributions",
    compile_as = [ "c++" ],
    srcs = glob(["source/distributions/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/distributions:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

# dal_example_suite(
#     name = "elastic_net",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/elastic_net/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/daal/src/algorithms/elastic_net:kernel",
#         "@onedal//cpp/daal/src/algorithms/optimization_solver:kernel",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS + [
#         "@tbb//:tbb",
#     ],
# )

# dal_example_suite(
#     name = "em",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/em/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/daal/src/algorithms/em:kernel",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS + [
#         "@tbb//:tbb",
#     ],
# )

# dal_example_suite(
#     name = "enable_thread_pinning",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/enable_thread_pinning/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS + [
#         "@tbb//:tbb",
#     ],
# )

# dal_example_suite(
#     name = "gradient_boosted_trees",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/gradient_boosted_trees/*.cpp"]),
#     dal_deps = [
#         "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS + [
#         "@tbb//:tbb",
#     ],
# )

dal_example_suite(
    name = "implicit_als",
    compile_as = [ "c++" ],
    srcs = glob(["source/implicit_als/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/implicit_als:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "k_nearest_neighbors",
    compile_as = [ "c++" ],
    srcs = glob(["source/k_nearest_neighbors/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/k_nearest_neighbors:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_algo_example_suite(
    algos = [
        "covariance",
        # "datasource",
        "dbscan",
        # "decision_forest",
        "kmeans",
        # "k_nearest_neighbors",
        # "kernel_function",
        "linear_regression",
        # "louvain",
        # "objective_function",
        # "pca",
        # "polynomial_kernel",
        # "rbf_kernel",
        # "shortest_paths",
        # "sigmoid_kernel",
        # "subgraph_isomorphism",
        # "svm",
        # "triangle_counting",
    ],
    compile_as = [ "c++" ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS,
)
