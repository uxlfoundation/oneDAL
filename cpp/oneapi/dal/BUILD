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
    # Mirror what Make puts on onedal.dll's Windows link line: the DAAL
    # implementation arrives through onedal_core's import library (`host_deps`
    # below), while the static MKL archives are linked in directly.
    #
    #   - Dropping the "daal" tag is what narrows the export surface. Those
    #     contexts carry DAAL *object files*, which the Windows path bundles
    #     into onedal.4_objects.lib and pulls in with /WHOLEARCHIVE, so their
    #     dllexport directives leak into onedal.dll's exports.
    #   - Keeping "mkl_embed" costs nothing in exports: it contributes
    #     prebuilt archives (mkl_intel_ilp64/mkl_tbb_thread/mkl_core), which
    #     stay outside the whole-archived object bundle and are searched only
    #     for unresolved references. Without it the CPU RNG kernels have no
    #     definition for MKL entry points such as `viRngUniform`.
    #     Make links the same archives via
    #     `daaldep.math_backend.shared_link_deps`.
    #
    # Linux is unchanged: libonedal.so resolves both DAAL and MKL symbols at
    # runtime via DT_NEEDED libonedal_core.so, so the host build needs neither
    # extra tag and the DPC build keeps its historical set. The Windows DPC
    # link takes MKL through the SYCL import libraries `@mkl//:mkl_dpc`
    # contributes, again matching Make.
    lib_tags = select({
        "@platforms//os:windows": ["dal", "mkl_embed"],
        "//conditions:default": ["dal"],
    }),
    # `BUILD_PARAMETERS_LIB` defaults to `no` on Windows, so Make folds the DPC
    # parameter objects straight into onedal_dpc.dll
    # (`ONEAPI.objs_y.dpc.lib += $(PARAMETERS.objs_y.dpc.filtered)`). Those
    # contexts are tagged "parameters", so the tag has to be accepted here or
    # the objects never reach the link and their dllexport'ed
    # `train_parameters_cpu` members are missing from the DLL's exports.
    # Linking `static_parameters_dpc` instead does not work: an archive is
    # searched only for unresolved references, while the export table is built
    # from the whole-archived object bundle.
    dpc_lib_tags = select({
        "@platforms//os:windows": ["dal", "parameters"],
        "//conditions:default": ["dal", "daal", "mkl_embed"],
    }),
    dal_deps = [
        ":core",
        ":optional",
    ],
    # On Windows, link onedal_core's import library plus parameter objects and
    # delay-load threading shims. Keep DAAL/MKL implementation in onedal_core.
    #   - core_dynamic: import library that resolves the DAAL algorithm and
    #     MKL backend symbols the oneAPI objects reference. `host_deps` is
    #     appended to both the host and the DPC link, so it must be listed
    #     here only -- repeating it in `dpc_deps` makes the label duplicate in
    #     `dynamic_dpc` and fails analysis.
    #   - static_parameters: parameter symbols not part of any other dep tree.
    #   - services_win_dll_shim: delay-load stubs for threading entry points
    #     (e.g. `_threaded_scalable_*`, `_daal_static_threader_reduce`) that
    #     the DAAL objects reference. The real TBB-backed symbols live in
    #     onedal_thread.dll. Mirrors onedal_core.dll above and the original
    #     Make build.
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
    # `host_deps` is appended to the host *and* the DPC link, so the shared
    # Windows inputs belong here rather than being split across `dpc_deps`.
    host_deps = select({
        "@platforms//os:windows": [
            "@onedal//cpp/daal:core_dynamic",
            "@onedal//cpp/daal:services_win_dll_shim",
        ],
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
