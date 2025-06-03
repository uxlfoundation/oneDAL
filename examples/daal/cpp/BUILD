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

def daal_example_suite(name, path, deps):
    dal_example_suite(
        name = name,
        compile_as = ["c++"],
        srcs = glob([f"source/{path}/*.cpp"]),
        dal_deps = deps,
        data = _DATA_DEPS,
        extra_deps = _TEST_DEPS + ["@tbb//:tbb"],
    )

daal_example_suite("association_rules", "association_rules", [
    "@onedal//cpp/daal/src/algorithms/assocrules:kernel",
])

daal_example_suite("boosting", "boosting", [
    "@onedal//cpp/daal/src/algorithms/adaboost:kernel",
    "@onedal//cpp/daal/src/algorithms/brownboost:kernel",
    "@onedal//cpp/daal/src/algorithms/logitboost:kernel",
])

daal_example_suite("cholesky", "cholesky", [
    "@onedal//cpp/daal/src/algorithms/cholesky:kernel",
])

daal_example_suite("decision_tree", "decision_tree", [
    "@onedal//cpp/daal/src/algorithms/decision_tree:kernel",
])

daal_example_suite("distance", "distance", [
    "@onedal//cpp/daal/src/algorithms/cordistance:kernel",
    "@onedal//cpp/daal/src/algorithms/cosdistance:kernel",
])

daal_example_suite("distributions", "distributions", [
    "@onedal//cpp/daal/src/algorithms/distributions:kernel",
])

daal_example_suite("elastic_net", "elastic_net", [
    "@onedal//cpp/daal/src/algorithms/elastic_net:kernel",
    "@onedal//cpp/daal/src/algorithms/optimization_solver:kernel",
])

daal_example_suite("em", "em", [
    "@onedal//cpp/daal/src/algorithms/em:kernel",
])

daal_example_suite("implicit_als", "implicit_als", [
    "@onedal//cpp/daal/src/algorithms/implicit_als:kernel",
])

daal_example_suite("k_nearest_neighbors", "k_nearest_neighbors", [
    "@onedal//cpp/daal/src/algorithms/k_nearest_neighbors:kernel",
])

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


dal_algo_example_suite(
    algos = [
        "covariance",
        # "datasource",
        "dbscan",
        "decision_forest",
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
