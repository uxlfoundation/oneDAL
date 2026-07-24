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
load("@onedal//dev/bazel/config:selects.bzl",
    "parameters_lib_disabled",
    "parameters_lib_enabled",
)

_PARAMETER_MODULES = [
    "@onedal//cpp/oneapi/dal/algo:parameters",
    "@onedal//cpp/oneapi/dal/detail/parameters",
]

_PARAMETER_MODULES_DPC = [
    "@onedal//cpp/oneapi/dal/algo:parameters_dpc",
    "@onedal//cpp/oneapi/dal/detail/parameters:parameters_dpc",
]

_DAL_LIB_TAGS = ["dal"] + parameters_lib_disabled(["parameters"])


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
    lib_tags = _DAL_LIB_TAGS,
    dal_deps = [
        ":core",
        ":optional",
    ],
    host_deps = parameters_lib_disabled(_PARAMETER_MODULES),
    dpc_deps = parameters_lib_disabled(_PARAMETER_MODULES_DPC),
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
    # extra tag. Only the non-Windows DPC branch keeps the historical
    # ["dal", "daal", "mkl_embed"] set; the Windows one is narrowed below. The
    # Windows DPC link takes MKL through the SYCL import libraries
    # `@mkl//:mkl_dpc` contributes, again matching Make.
    lib_tags = select({
        "@platforms//os:windows": ["dal", "mkl_embed"],
        "//conditions:default": ["dal"],
    }) + parameters_lib_disabled(["parameters"]),
    # `BUILD_PARAMETERS_LIB` defaults to `no` on Windows, so Make folds the DPC
    # parameter objects straight into onedal_dpc.dll
    # (`ONEAPI.objs_y.dpc.lib += $(PARAMETERS.objs_y.dpc.filtered)`). Those
    # contexts are tagged "parameters", so the tag has to be accepted here or
    # the objects never reach the link and their dllexport'ed
    # `train_parameters_cpu` members are missing from the DLL's exports.
    # Linking `static_parameters_dpc` instead does not work: an archive is
    # searched only for unresolved references, while the export table is built
    # from the whole-archived object bundle.
    #
    # Linux must *not* accept the tag. There `BUILD_PARAMETERS_LIB` is `yes`, so
    # Make emits the parameters as their own shared object
    # (`parameters_y := $(plib)onedal_parameters...`, makefile:343) and
    # libonedal_dpc.so never contains those objects. Adding the tag here would
    # fold them into onedal_dpc, duplicating the definitions and diverging from
    # the Make release the comparator checks against.
    dpc_lib_tags = select({
        "@platforms//os:windows": ["dal", "parameters"],
        "//conditions:default": ["dal", "daal", "mkl_embed"],
    }) + parameters_lib_disabled(["parameters"]),
    dal_deps = [
        ":core",
        ":optional",
    ],
    dpc_deps = parameters_lib_disabled(_PARAMETER_MODULES_DPC),
    # On Windows, link onedal_core's import library plus parameter objects and
    # delay-load threading shims. Keep DAAL/MKL implementation in onedal_core.
    # `host_deps` is appended to both the host and the DPC link, so shared
    # Windows dependencies belong here rather than being duplicated.
    host_deps = parameters_lib_disabled(_PARAMETER_MODULES) + select({
        "@platforms//os:windows": [
            "@onedal//cpp/daal:core_dynamic",
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
    ] + parameters_lib_enabled([
        ":static_parameters",
        ":static_parameters_dpc",
    ]),
)

filegroup(
    name = "all_dynamic",
    srcs = [
        ":dynamic",
        ":dynamic_dpc",
    ] + parameters_lib_enabled([
        ":dynamic_parameters",
        ":dynamic_parameters_dpc",
    ]),
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
