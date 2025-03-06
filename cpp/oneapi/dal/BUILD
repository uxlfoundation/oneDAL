package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_collect_modules",
    "dal_public_includes",
    "dal_static_lib",
    "dal_dynamic_lib",
    "dal_test_suite",
    "dal_collect_test_suites",
    "dal_generate_cpu_dispatcher",
)

dal_generate_cpu_dispatcher(
    name = "cpu_dispatcher",
    out = "_dal_cpu_dispatcher_gen.hpp",
)

dal_module(
    name = "common",
    auto = True,
    hdrs = [
        ":cpu_dispatcher",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi:dal_header",
    ],
    extra_deps = [
        "@onedal//cpp/oneapi:include_root",
        "@onedal//cpp/daal:services",
        "@onedal//cpp/daal:data_management",
    ],
    dpc_deps = [
        "@mkl//:mkl_dpc",
        "@dpl//:headers",
    ],
)

dal_collect_modules(
    name = "core",
    root = "@onedal//cpp/oneapi/dal",
    modules = [
        "graph",
        "table",
        "util",
    ],
    dal_deps = [
        ":common",
    ],
)

dal_collect_modules(
    name = "optional",
    root = "@onedal//cpp/oneapi/dal",
    modules = [
        "algo",
        "io",
        "backend/primitives",
    ],
)

dal_public_includes(
    name = "public_includes",
    dal_deps = [
        ":core",
        ":optional",
        "@onedal//cpp/oneapi/dal/detail/mpi",
        "@onedal//cpp/oneapi/dal/detail/ccl",
        "@onedal//cpp/oneapi/dal/detail/parameters",
        "@onedal//cpp/oneapi/dal/algo:parameters",
    ],
)

dal_static_lib(
    name = "static",
    lib_name = "onedal",
    dal_deps = [
        ":core",
        ":optional",
    ],
)

dal_static_lib(
    name = "static_parameters",
    lib_name = "onedal_parameters",
    dal_deps = [
        ":static",
        "@onedal//cpp/oneapi/dal/algo:parameters",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
)

dal_dynamic_lib(
    name = "dynamic",
    lib_name = "onedal",
    dal_deps = [
        ":core",
        ":optional",
    ],
)

dal_dynamic_lib(
    name = "dynamic_parameters",
    lib_name = "onedal_parameters",
    dal_deps = [
        ":dynamic",
        "@onedal//cpp/oneapi/dal/algo:parameters",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
)

filegroup(
    name = "all_static",
    srcs = [
        ":static",
        ":static_dpc",
        ":static_parameters",
        ":static_parameters_dpc",
    ],
)

filegroup(
    name = "all_dynamic",
    srcs = [
        ":dynamic",
        ":dynamic_dpc",
        ":dynamic_parameters",
        ":dynamic_parameters_dpc",
    ],
)

dal_test_suite(
    name = "common_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ], exclude=[
        "test/mpi_*.cpp",
        "test/ccl_*.cpp"
    ]),
    dal_deps = [ ":common" ],
)

dal_test_suite(
    name = "mpi_communicator_test",
    mpi = True,
    mpi_ranks = 2,
    framework = "catch2",
    srcs = glob([
        "test/mpi_*.cpp"
    ]),
    dal_deps = [
        ":common",
    ],
)

dal_test_suite(
    name = "ccl_communicator_test",
    ccl = True,
    mpi_ranks = 2,
    framework = "catch2",
    srcs = glob([
        "test/ccl_*.cpp"
    ]),
    dal_deps = [
        ":common",
    ],
)

dal_collect_test_suites(
    name = "tests",
    root = "@onedal//cpp/oneapi/dal",
    modules = [
        "algo",
        "graph",
        "io",
        "table",
        "util",
        "backend/primitives",
    ],
    tests = [
        ":common_tests",
        # TODO: Temporary disabled due to
        #       unexpectedly high build time
        # "@onedal//cpp/oneapi:dal_hpp_test",
    ],
)

dal_collect_test_suites(
    name = "mpi_tests",
    root = "@onedal//cpp/oneapi/dal",
    tests = [
        "@onedal//cpp/oneapi/dal/algo/kmeans:mpi_tests",
        "@onedal//cpp/oneapi/dal/algo/dbscan:mpi_tests",
    ],
)

dal_collect_test_suites(
    name = "ccl_tests",
    root = "@onedal//cpp/oneapi/dal",
    tests = [
        "@onedal//cpp/oneapi/dal/algo/kmeans:ccl_tests",
        "@onedal//cpp/oneapi/dal/algo/dbscan:ccl_tests",
    ],
)
