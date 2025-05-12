package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "backend",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/pca/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/pca/backend/gpu",
    ],
)

dal_module(
    name = "common",
    hdrs = glob(["*.hpp"]),
)
