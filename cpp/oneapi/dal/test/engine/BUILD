package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_test_module",
)

dal_test_module(
    name = "common",
    auto = True,
    auto_exclude = [
        "catch_main.cpp",
        "mpi*.*",
        "ccl*.*",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal:table",
        "@onedal//cpp/oneapi/dal/io:csv",
        "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
    ],
    dal_test_deps = [
        "@onedal//cpp/oneapi/dal/test/engine/linalg",
        "@onedal//cpp/oneapi/dal/test/engine/metrics",
    ],
    extra_deps = [
        "@boost//:boost",
        "@catch2//:catch2",
        "@eigen//:eigen",
        "@fmt//:fmt",
    ],
)

dal_test_module(
    name = "mpi",
    hdrs = [
        "mpi_global.hpp",
    ],
    srcs = [
        "mpi_global.cpp",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal/detail/mpi",
    ],
    dal_test_deps = [
        ":common",
    ],
)

dal_test_module(
    name = "ccl",
    hdrs = [
        "ccl_global.hpp",
    ],
    srcs = [
        "ccl_global.cpp",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal/detail/ccl",
    ],
    dal_test_deps = [
        ":common",
    ],
)

dal_test_module(
    name = "catch2_main",
    srcs = [
        "catch_main.cpp",
    ],
    dal_test_deps = [
        ":common",
    ],
    extra_deps = [
        "@catch2//:catch2",
    ],
)
