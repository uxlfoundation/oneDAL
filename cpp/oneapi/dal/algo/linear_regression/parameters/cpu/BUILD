package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "cpu",
    auto = True,
    lib_tag = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/linear_regression:core",
    ],
)
