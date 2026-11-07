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
        "@onedal//cpp/oneapi/dal/algo/linear_regression/backend:model_impl",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/linear_regression/parameters",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
)

dal_module(
    name = "linear_regression",
    dal_deps = [
        ":core",
        ":parameters",
        "@onedal//cpp/oneapi/dal/algo/linear_regression/detail",
        "@onedal//cpp/oneapi/dal/algo/linear_regression/backend",
    ]
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    hdrs = glob([
        "test/*.hpp",
    ], exclude=[
        "test/mpi_*.hpp",
        "test/ccl_*.hpp"
    ]),
    srcs = glob([
        "test/*.cpp",
    ], exclude=[
        "test/mpi_*.cpp",
        "test/ccl_*.cpp",
        "test/online_spmd.cpp",
    ]),
    dal_deps = [
        ":linear_regression",
    ],
)

# Host SPMD finalize is not implemented, so this test only runs on a device.
dal_test_suite(
    name = "online_spmd_tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    hdrs = glob([
        "test/*.hpp",
    ], exclude=[
        "test/mpi_*.hpp",
        "test/ccl_*.hpp",
    ]),
    srcs = [
        "test/online_spmd.cpp",
    ],
    dal_deps = [
        ":linear_regression",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
        ":online_spmd_tests",
    ],
)
