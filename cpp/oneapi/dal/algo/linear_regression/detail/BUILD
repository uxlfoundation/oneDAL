load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "detail",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/linear_regression:core",
        "@onedal//cpp/oneapi/dal/algo/linear_regression/backend",
        "@onedal//cpp/oneapi/dal/algo/linear_regression/parameters",
    ],
)
