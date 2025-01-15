package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "core",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/covariance/parameters",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
)

dal_module(
    name = "covariance",
    dal_deps = [
        ":core",
        ":parameters",
        "@onedal//cpp/oneapi/dal/algo/covariance/detail",
        "@onedal//cpp/oneapi/dal/algo/covariance/backend",
    ]
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    hdrs = glob([
        "test/*.hpp",
    ]),
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":covariance",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
    ],
)
