load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "parameters",
    auto = True,
    lib_tag = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
    ]
)
