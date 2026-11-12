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
    "parameters_lib_target_compatible_with",
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
    # Make passes --offload-compress to the DPC link of this library
    # (makefile:821, LOPT). Host links do not get it.
    dpc_linkopts = ["--offload-compress"],
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
    # Linux needs "mkl_embed" for the same reason: the CPU RNG kernels in
    # libonedal.so reference MKL entry points directly, and the copy of MKL
    # inside libonedal_core.so cannot satisfy them -- it is localized by
    # --exclude-libs, so `viRngUniform` is a local `t` symbol there and never
    # reaches the dynamic symbol table. Without the tag libonedal.so ships
    # with `viRngUniform` undefined and no DT_NEEDED that can provide it, so
    # linking it into a consumer fails unless that consumer happens to put MKL
    # on its own link line. Make has always linked the same archives into
    # libonedal.so via `daaldep.math_backend.shared_link_deps` (its
    # libonedal.so carries `t viRngUniform` too), so this matches the Make
    # release rather than changing behaviour. Exports are unaffected: the
    # archives are searched only for unresolved references, and the dynamic
    # symbol table has the same 3588 entries with and without the tag.
    #
    # The non-Windows DPC branch below already carries the historical
    # ["dal", "daal", "mkl_embed"] set; the Windows one is narrowed. The
    # Windows DPC link takes MKL through the SYCL import libraries
    # `@mkl//:mkl_dpc` contributes, again matching Make.
    lib_tags = select({
        "@platforms//os:windows": ["dal", "mkl_embed"],
        "//conditions:default": ["dal", "mkl_embed"],
    }) + parameters_lib_disabled(["parameters"]),
    # Windows accepts the "parameters" tag: Make folds those objects into onedal_dpc.dll; a static archive would not export them.
    # Linux drops it and the daal/mkl_embed tags: Make links dynamic MKL and keeps parameters in their own .so.
    dpc_lib_tags = select({
        "@platforms//os:windows": ["dal", "parameters"],
        "//conditions:default": ["dal"],
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
    }) + select({
        # Same reason the "mkl_embed" tag above is kept for the MKL backend: the
        # oneAPI CPU kernels reference the math backend directly, and a Windows
        # DLL has to resolve those references at link time. OpenBLAS is not
        # tagged "mkl_embed", so name it here. `openblas.lib` is a static
        # archive, so this embeds the BLAS/LAPACK entry points the oneAPI
        # kernels use, the way `mkl_embed` does for the MKL backend.
        "@config//:backend_ref_windows": ["@openblas//:openblas"],
        "//conditions:default": [],
    }),
    dpc_host_deps = select({
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
    # Make passes --offload-compress to the DPC link of this library
    # (makefile:821, LOPT). Host links do not get it.
    dpc_linkopts = ["--offload-compress"],
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
    dpc_host_deps = select({
        "@platforms//os:windows": [
            "@onedal//cpp/daal:core_dynamic",
            "@onedal//cpp/daal:services_win_dll_shim",
        ],
        "//conditions:default": [],
    }),
    target_compatible_with = parameters_lib_target_compatible_with(),
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
