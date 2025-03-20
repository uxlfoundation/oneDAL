package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "correlation_distance",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ]
)

dal_test_suite(
    name = "tests",
)