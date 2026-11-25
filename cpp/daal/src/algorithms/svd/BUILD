package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:public_includes_svd",
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal/src/algorithms/qr:kernel",
    ],
)
