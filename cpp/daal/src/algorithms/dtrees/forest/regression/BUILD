load("@onedal//dev/bazel:daal.bzl", "daal_module")

package(default_visibility = ["//visibility:public"])

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
        "@onedal//cpp/daal/src/algorithms/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest:kernel",
    ],
)
