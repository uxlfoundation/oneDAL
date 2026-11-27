load("@onedal//dev/bazel:daal.bzl", "daal_module")
load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_collect_test_suites",
)

package(default_visibility = ["//visibility:public"])

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
    ],
)

dal_collect_test_suites(
    name = "tests",
    root = "@onedal//cpp/daal/src/algorithms/dtrees/gbt",
    modules = [
        "regression",
    ],
)
