package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "kmeans",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:selection",
        "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
        "@onedal//cpp/oneapi/dal/backend/primitives:distance",
        "@onedal//cpp/oneapi/dal/backend/primitives:sort",
        "@onedal//cpp/oneapi/dal/backend/primitives:sparse_blas",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/kmeans:kernel",
    ],
)

dal_test_suite(
    name = "cpu_tests",
    private = True,
    compile_as = [ "c++" ],
    srcs = glob([
        "backend/cpu/test/*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "gpu_tests",
    framework = "catch2",
    private = True,
    compile_as = [ "dpc++" ],
    srcs = glob([
        "backend/gpu/test/*.cpp",
    ], exclude=[
        "backend/gpu/test/perf_*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
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
        ":kmeans",
    ],
)

dal_test_suite(
    name = "mpi_tests",
    mpi = True,
    mpi_ranks = 2,
    framework = "catch2",
    hdrs = glob([
        "test/mpi_*.hpp",
        "test/data.hpp",
        "test/spmd_backend_fixture.hpp",
        "test/spmd_backend_template.hpp",
    ]),
    srcs = glob([
        "test/mpi_*.cpp"
    ]),
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "ccl_tests",
    ccl = True,
    mpi_ranks = 2,
    framework = "catch2",
    hdrs = glob([
        "test/ccl_*.hpp",
        "test/data.hpp",
        "test/spmd_backend_fixture.hpp",
        "test/spmd_backend_template.hpp",
    ]),
    srcs = glob([
        "test/ccl_*.cpp"
    ]),
    dal_deps = [
        ":kmeans",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":cpu_tests",
        ":gpu_tests",
        ":interface_tests",
    ],
)

dal_test_suite(
    name = "gpu_perf_tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    private = True,
    srcs = glob([
        "backend/gpu/test/perf_*.cpp",
    ]),
    dal_deps = [
        ":kmeans",
    ],
)
