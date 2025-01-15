package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "dbscan",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:selection",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/dbscan:kernel",
    ],
)

dal_test_suite(
    name = "mpi_tests",
    mpi = True,
    mpi_ranks = 2,
    framework = "catch2",
    hdrs = glob([
        "test/data.hpp",
        "test/spmd_backend_fixture.hpp",
        "test/spmd_backend_template.hpp",
    ]),
    srcs = glob([
        "test/mpi_*.cpp"
    ]),
    dal_deps = [
        ":dbscan",
    ],
)

dal_test_suite(
    name = "ccl_tests",
    ccl = True,
    mpi_ranks = 2,
    framework = "catch2",
    hdrs = glob([
        "test/data.hpp",
        "test/spmd_backend_fixture.hpp",
        "test/spmd_backend_template.hpp",
    ]),
    srcs = glob([
        "test/ccl_*.cpp"
    ]),
    dal_deps = [
        ":dbscan",
    ],
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
        "test/ccl_*.cpp"
    ]),
    dal_deps = [
        ":dbscan",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
    ],
)
