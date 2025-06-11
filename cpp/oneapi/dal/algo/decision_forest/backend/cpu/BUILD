package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "cpu",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/decision_forest:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
    extra_deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/classification:kernel",
    ],
)
