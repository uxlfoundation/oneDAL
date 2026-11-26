load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "newton_cg",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
    ],
)
