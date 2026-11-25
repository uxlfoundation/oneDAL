package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:public_includes_lasso_regression",
        "@onedal//cpp/daal:public_includes_optimization_solver",
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal/src/algorithms/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/linear_model:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function/mse:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver:kernel",
        "@onedal//cpp/daal/src/algorithms/optimization_solver/coordinate_descent:kernel",
    ],
)
