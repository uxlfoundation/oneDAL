package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
)

dal_module(
    name = "gpu",
    auto = True,
    lib_tag = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/decision_forest:core",
    ],
)
