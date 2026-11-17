package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:public_includes_pca",
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
        "@onedal//cpp/daal/src/algorithms/covariance:kernel",
        "@onedal//cpp/daal/src/algorithms/svd:kernel",
        "@onedal//cpp/daal/src/algorithms/normalization/zscore:kernel",
    ],
)
