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
    lib_tags = ["parameters"],
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo:parameters",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
)

dal_dynamic_lib(
    name = "dynamic",
    lib_name = "onedal",
    # Both Make and Bazel consume DAAL/MKL implementation through onedal_core's
    # import library. Selecting those tags here pulls their objects into
    # onedal.dll via /WHOLEARCHIVE and widens its export surface.
    lib_tags = ["dal"],
    dpc_lib_tags = ["dal"],
    dal_deps = [
        ":core",
        ":optional",
    ],
    dpc_deps = select({
        "@platforms//os:windows": [
            "@onedal//cpp/daal:core_dynamic",
            ":static_parameters_dpc",
        ],
        "//conditions:default": [],
    }),
    # On Windows, link onedal_core's import library plus parameter objects and
    # delay-load threading shims. Keep DAAL/MKL implementation in onedal_core.
    #   - static_parameters: parameter symbols not part of any other dep tree.
    #   - services_win_dll_shim: delay-load stubs for threading entry points
    #     (e.g. `_threaded_scalable_*`, `_daal_static_threader_reduce`) that
    #     the embedded DAAL kernel objects reference. The real TBB-backed
    #     symbols live in onedal_thread.dll. Mirrors onedal_core.dll above
    #     and the original Make build.
    host_deps = select({
        "@platforms//os:windows": [
            "@onedal//cpp/daal:core_dynamic",
            ":static_parameters",
            "@onedal//cpp/daal:services_win_dll_shim",
        ],
        "//conditions:default": [],
    }),
)

dal_dynamic_lib(
    name = "dynamic_parameters",
    lib_name = "onedal_parameters",
    # On Windows, DAAL/MKL stay behind onedal_core's import library; only
    # parameter and oneAPI objects belong in this DLL. Keep Linux unchanged.
    lib_tags = select({
        "@platforms//os:windows": ["parameters", "dal"],
        "//conditions:default": ["parameters"],
    }),
    dpc_lib_tags = select({
        "@platforms//os:windows": ["parameters", "dal"],
        "//conditions:default": ["parameters"],
    }),
    dal_deps = [
        ":dynamic",
        "@onedal//cpp/oneapi/dal/algo:parameters",
        # The logistic regression parameter objects instantiate default
        # optimizer wrappers implemented in the algorithm detail module. Linux
        # keeps this resolved through libonedal.so, but Windows DLL links must
        # see the implementation directly when building onedal_parameters.dll.
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/detail",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
    dpc_deps = select({
        "@platforms//os:windows": [
            "@onedal//cpp/daal:core_dynamic",
        ],
        "//conditions:default": [],
    }),
    host_deps = select({
        "@platforms//os:windows": ["@onedal//cpp/daal:services_win_dll_shim"],
        "//conditions:default": [],
    }),
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
