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

dal_example_suite(
    name = "datasource",
    compile_as = [ "c++" ],
    srcs = glob(["source/datasource/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/low_order_moments:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "decision_forest",
    compile_as = [ "c++" ],
    srcs = glob(["source/decision_forest/*.cpp"],
    exclude = ["source/decision_forest/df_cls_traversed_model_builder.cpp"],),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/decision_tree:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/classification:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/regression:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "decision_tree",
    compile_as = [ "c++" ],
    srcs = glob(["source/decision_tree/*.cpp"],
    exclude = ["source/decision_tree/dt_cls_dense_batch.cpp",
    "source/decision_tree/dt_reg_dense_batch.cpp"],),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/decision_tree:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/classification:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/regression:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

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

dal_example_suite(
    name = "elastic_net",
    compile_as = [ "c++" ],
    srcs = glob(["source/elastic_net/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/elastic_net:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/adagrad:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/coordinate_descent:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/lbfgs:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/saga:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/sgd:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function/mse:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "em",
    compile_as = [ "c++" ],
    srcs = glob(["source/em/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/em:kernel",
        "@onedal//cpp/daal/src/algorithms/covariance:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "enable_thread_pinning_example",
    compile_as = [ "c++" ],
    srcs = glob(["source/enable_thread_pinning/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "gradient_boosted_trees",
    compile_as = [ "c++" ],
    srcs = glob(["source/gradient_boosted_trees/*.cpp"],
    exclude = ["source/gradient_boosted_trees/gbt_cls_traversed_model_builder.cpp",
    "source/gradient_boosted_trees/gbt_reg_traversed_model_builder.cpp"],),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/decision_tree:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/classification:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt/classification:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

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

dal_example_suite(
    name = "kernel_function",
    compile_as = [ "c++" ],
    srcs = glob(["source/kernel_function/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/kernel_function:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "lasso_regression",
    compile_as = [ "c++" ],
    srcs = glob(["source/lasso_regression/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/lasso_regression:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/adagrad:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/coordinate_descent:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/lbfgs:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/saga:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/sgd:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function/mse:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "logistic_regression",
    compile_as = [ "c++" ],
    srcs = glob(["source/logistic_regression/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/logistic_regression:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/adagrad:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/coordinate_descent:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/lbfgs:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/saga:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/sgd:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function/mse:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "moments",
    compile_as = [ "c++" ],
    srcs = glob(["source/moments/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/low_order_moments:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "naive_bayes",
    compile_as = [ "c++" ],
    srcs = glob(["source/naive_bayes/*.cpp"],
    exclude = ["source/naive_bayes/mn_naive_bayes_csr_batch.cpp"],),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/naivebayes:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

# dal_example_suite(
#     name = "normalization",
#     compile_as = [ "c++" ],
#     srcs = glob(["source/normalization/*.cpp"],
#     exclude = ["source/normalization/minmax_dense_batch.cpp", "source/normalization/zscore_dense_batch.cpp"],),
#     dal_deps = [
#         "@onedal//cpp/daal/src/algorithms/normalization/minmax:kernel",
#         "@onedal//cpp/daal/src/algorithms/normalization/zscore:kernel",
#     ],
#     data = _DATA_DEPS,
#     extra_deps = _TEST_DEPS + [
#         "@tbb//:tbb",
#     ],
# )

dal_example_suite(
    name = "optimization_solvers",
    compile_as = [ "c++" ],
    hdrs = glob(["source/optimization_solvers/*.h"]),
    srcs = glob(["source/optimization_solvers/*.cpp"],
    exclude = ["source/optimization_solvers/saga_logistic_loss_dense_batch.cpp"],),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/optimization_solver:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/adagrad:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/coordinate_descent:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/lbfgs:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/saga:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/sgd:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function/mse:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function/cross_entropy_loss:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function/logistic_loss:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "outlier_detection",
    compile_as = [ "c++" ],
    srcs = glob(["source/outlier_detection/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/outlierdetection_bacon:kernel",
        "@onedal//cpp/daal/src/algorithms/outlierdetection_multivariate:kernel",
        "@onedal//cpp/daal/src/algorithms/outlierdetection_univariate:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)

dal_example_suite(
    name = "pca_transform",
    compile_as = [ "c++" ],
    srcs = glob(["source/pca_transform/*.cpp"]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/pca:kernel",
        "@onedal//cpp/daal/src/algorithms/pca/transform:kernel",
    ],
    data = _DATA_DEPS,
    extra_deps = _TEST_DEPS + [
        "@tbb//:tbb",
    ],
)


dal_algo_example_suite(
    algos = [
        "covariance",
        "dbscan",
        # "decision_forest",
        "kmeans",
        # "k_nearest_neighbors",
        # "kernel_function",
        "linear_regression",
        # "louvain",
        # "objective_function",
        "pca",
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
