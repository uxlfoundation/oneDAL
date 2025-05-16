package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/pca/parameters/cpu",
        "@onedal//cpp/oneapi/dal/algo/pca/parameters/gpu",
    ],
)
