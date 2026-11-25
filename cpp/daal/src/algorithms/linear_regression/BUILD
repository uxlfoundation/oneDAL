package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:public_includes_linear_model",
        "@onedal//cpp/daal:public_includes_linear_regression",
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
        "@onedal//cpp/daal/src/algorithms/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/linear_model:kernel",
    ],
)
