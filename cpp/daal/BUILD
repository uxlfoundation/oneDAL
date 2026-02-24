package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_test_suite",
    "dal_collect_test_suites",
)
load("@onedal//dev/bazel:daal.bzl",
    "daal_module",
    "daal_static_lib",
    "daal_dynamic_lib",
    "daal_algorithms",
    "daal_generate_version",
    "daal_patch_kernel_defines",
)
load("@rules_shell//shell:sh_test.bzl", "sh_test")

daal_module(
    name = "microvmlipp",
    lib_tag = "mkl_embed",
    override_deps_lib_tag = True,
    deps = select({
        "@config//:backend_ref": [ ],
        "//conditions:default": [
                                    "@mkl//:mkl_core",
                                ],
        }),
)

daal_module(
    name = "mathbackend_thread",
    lib_tag = select({
        "@config//:backend_ref": "daal",
        "//conditions:default": "mkl_embed",
    }),
    override_deps_lib_tag = select({
        "@config//:backend_ref": False,
        "//conditions:default": True,
    }),
    deps = select({
        "@config//:backend_ref": [ "@openblas//:openblas",
                                 ],
        "//conditions:default": [ "@mkl//:mkl_core",
                                ],
        }),
)

daal_patch_kernel_defines(
    name = "kernel_defines",
    src = "include/services/internal/daal_kernel_defines.h",
)

daal_module(
    name = "public_includes",
    hdrs = glob([
        "include/**/*.h",
    ], exclude = [
        "**/daal_kernel_defines.h",
        "**/daal_win.h",
    ]),
    includes = [ "include" ],
)

daal_module(
    name = "includes",
    hdrs = [ ":kernel_defines",
    ],
    includes = [ "." ],
    defines = [
        "DAAL_NOTHROW_EXCEPTIONS",
        "DAAL_HIDE_DEPRECATED",
    ],
    deps = select({
        "@config//:backend_ref": [
                                   ":public_includes",
                                   "@openblas//:headers",
                                 ],
        "//conditions:default": [
                                  ":public_includes",
                                  "@mkl//:headers",
                                ],
        }),
)

daal_generate_version(
    name = "version",
    out = "daal_version_gen.h",
)

daal_module(
    name = "service_headers",
    hdrs = glob([
        "src/services/**/*.h",
        "src/externals/**/*.h",
        "src/algorithms/*.h",
    ]) + [
        ":version",
    ],
    deps = [
        ":includes",
    ],
)

daal_module(
    name = "threading_headers",
    hdrs = glob(["src/threading/**/*.h"]),
    deps = [
        ":service_headers",
    ],
)

daal_module(
    name = "services",
    srcs = glob([
            "src/services/**/*.cpp",
            "src/externals/**/*.cpp",
            "src/algorithms/*.cpp",
        ],
        exclude = ["src/externals/**/*_win_dll.cpp"],
    ),
    deps = [
        ":service_headers",
        ":threading_headers",
        ":microvmlipp",
    ],
)

# Windows-only shim that defines `_threaded_scalable_{malloc,free,...}`
# and `_daal_static_threader_reduce` entry points, which delay-load
# `onedal_thread.dll` at runtime. The Makefile pulls
# `core_threading_win_dll.o` into the dynamic core on Windows only
# (makefile:507-509). Linux resolves these symbols against
# `libonedal_thread.so` at executable link, so the shim is unused there.
# Each Windows DLL that holds DAAL algorithm code needs its own copy of
# this shim (LNK2001 if missing, but separate output binaries never
# collide on LNK1169 because each DLL gets its own .obj set).
daal_module(
    name = "services_win_dll_shim",
    srcs = select({
        "@platforms//os:windows": glob(["src/externals/**/*_win_dll.cpp"]),
        "//conditions:default": [],
    }),
    deps = [
        ":service_headers",
        ":threading_headers",
    ],
)

daal_module(
    name = "data_management",
    hdrs = glob(["src/data_management/**/*.h", "src/data_management/**/*.i"]),
    srcs = glob(["src/data_management/**/*.cpp"]),
    deps = [
        ":services",
    ],
)

daal_module(
    name = "threading_tbb",
    srcs = glob(["src/threading/**/*.cpp"]),
    local_defines = [
        "__TBB_NO_IMPLICIT_LINKAGE",
        "__TBB_LEGACY_MODE",
        "TBB_SUPPRESS_DEPRECATED_MESSAGES",
        "TBB_USE_ASSERT=0",
    ],
    deps = select({
            "@config//:backend_ref": [
                                          ":threading_headers",
                                          ":mathbackend_thread",
                                          "@tbb//:tbb",
                                          "@tbb//:tbbmalloc",
                                        ],
            "//conditions:default": [
                                      ":threading_headers",
                                      ":mathbackend_thread",
                                      "@tbb//:tbb",
                                      "@tbb//:tbbmalloc",
                                    ],
        }),
)

daal_module(
    name = "engines",
    deps = [
        "@onedal//cpp/daal/src/algorithms/engines:kernel",
    ],
)

daal_module(
    name = "core",
    deps = [
        ":services",
        ":data_management",
    ],
)

daal_algorithms(
    name = "all_algorithms",
    algorithms = [
        "cholesky",
        "classifier",
        "cordistance",
        "cosdistance",
        "covariance",
        "dbscan",
        "distributions",
        "dtrees",
        "dtrees/forest/classification",
        "dtrees/forest/regression",
        "dtrees/gbt/regression",
        "dtrees/gbt/classification",
        "elastic_net",
        "em",
        "engines",
        "implicit_als",
        "k_nearest_neighbors",
        "kernel_function",
        "kmeans",
        "lasso_regression",
        "linear_model",
        "linear_regression",
        "logistic_regression",
        "low_order_moments",
        "multiclassclassifier",
        "normalization/minmax",
        "normalization/zscore",
        "objective_function",
        "objective_function/cross_entropy_loss",
        "objective_function/logistic_loss",
        "objective_function/mse",
        "optimization_solver",
        "optimization_solver/lbfgs",
        "optimization_solver/coordinate_descent",
        "pca",
        "pca/transform",
        "pivoted_qr",
        "qr",
        "regression",
        "ridge_regression",
        "svd",
        "svm",
    ],
)

daal_static_lib(
    name = "core_static",
    lib_name = "onedal_core",
    deps = [
        ":core",
        ":all_algorithms",
    ],
)


daal_static_lib(
    name = "thread_static",
    lib_name = "onedal_thread",
    deps = [
        ":threading_tbb",
    ],
)

daal_dynamic_lib(
    name = "core_dynamic",
    lib_name = "onedal_core",
    deps = [
        ":core",
        ":all_algorithms",
    ] + select({
        # MSVC requires all DLL symbols to be resolved at link time. On
        # Windows the threading entry points come from
        # `services_win_dll_shim` (which delay-loads onedal_thread.dll)
        # rather than from `:threading_tbb` directly — pulling in
        # threading_tbb here would duplicate every threading symbol with
        # the shim and trigger LNK2005. Mirrors makefile:507-509: only
        # `core_threading_win_dll.o` lands in the dynamic core; the real
        # TBB-backed threading objects live in onedal_thread.dll.
        "@platforms//os:windows": [":services_win_dll_shim"],
        "//conditions:default": [],
    }),
)

daal_dynamic_lib(
    name = "thread_dynamic",
    lib_name = "onedal_thread",
    deps = [
        ":threading_tbb",
    ],
)

daal_module(
    name = "threading_static",
    deps = select({
        "@config//:test_thread_mode_par": [
            "@onedal//cpp/daal:thread_static",
        ]
    }),
)

daal_module(
    name = "threading_dynamic",
    deps = select({
        "@config//:test_thread_mode_par": [
            "@onedal//cpp/daal:thread_dynamic",
        ]
    }),
)

daal_module(
    name = "threading_release_static",
    deps = select({
        "@config//:test_thread_mode_par": [
            "@onedal_release//:thread_static",
        ]
    }),
)

daal_module(
    name = "threading_release_dynamic",
    deps = select({
        "@config//:test_thread_mode_par": [
            "@onedal_release//:thread_dynamic",
        ]
    }),
)

filegroup(
    name = "all_static",
    srcs = [
        ":core_static",
        ":thread_static",
    ],
)

dal_collect_test_suites(
    name = "tests",
    root = "@onedal//cpp/daal/src/algorithms",
    modules = [
        "dtrees/gbt/regression"
    ],
)

sh_test(
    name = "mkl_linkage_test",
    srcs = ["//dev/bazel/tests:mkl_linkage_test.sh"],
    data = [
        ":core_static",
        ":core_dynamic",
    ],
    # Only meaningful with MKL backend; skip for OpenBLAS/reference backend.
    # Also Linux-only: uses GNU nm -D and .so outputs.
    target_compatible_with = select({
        "@config//:backend_ref": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }) + ["@platforms//os:linux"],
)

sh_test(
    name = "isa_coverage_test",
    srcs = ["@onedal//dev/bazel/tests:isa_coverage_test.sh"],
    args = ["$(location :core_dynamic)"],
    data = [":core_dynamic"],
    tags = ["manual"],
)
