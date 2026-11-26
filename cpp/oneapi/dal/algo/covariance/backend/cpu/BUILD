load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "cpu",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/covariance:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
    extra_deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal/src/algorithms/covariance:kernel",
    ],
)
