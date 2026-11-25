package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:public_includes_classifier",
        "@onedal//cpp/daal:public_includes_engines",
        "@onedal//cpp/daal:public_includes_k_nearest_neighbors",
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
        "@onedal//cpp/daal/src/algorithms/classifier:kernel",
    ],
)
