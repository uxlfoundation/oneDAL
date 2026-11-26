load("@onedal//dev/bazel:daal.bzl", "daal_module")

package(default_visibility = ["//visibility:public"])

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal/src/algorithms/covariance:kernel",
        "@onedal//cpp/daal/src/algorithms/distributions:kernel",
    ],
)
