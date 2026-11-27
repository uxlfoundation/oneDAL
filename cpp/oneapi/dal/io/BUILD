load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_collect_modules",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

IOS = [
    "csv",
]

dal_collect_modules(
    name = "io",
    root = "@onedal//cpp/oneapi/dal/io",
    modules = IOS,
)

dal_test_suite(
    name = "tests",
    tests = [],
)
