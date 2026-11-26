load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/parameters/cpu",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/parameters/gpu",
    ],
)
