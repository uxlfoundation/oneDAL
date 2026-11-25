package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/oneapi/**/*.hpp",
    ], allow_empty = True),
    includes = [ "include" ],
)

cc_library(
    name = "core_static",
    srcs = [
        "lib/%{arch_dir}/libonedal_core.a",
    ],
    deps = [
        ":headers",
        # TODO: drop TBB once vml_ipp no longer depends on it.
        "@tbb//:tbb_binary",
        "@mkl//:mkl_static",
    ],
)

cc_library(
    name = "thread_static",
    srcs = [
        "lib/%{arch_dir}/libonedal_thread.a",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
    ],
)

cc_library(
    name = "onedal_static",
    srcs = [
        "lib/%{arch_dir}/libonedal.a",
    ] + glob([
        # Separate-layout packages only; when absent, the package is folded, not broken.
        "lib/%{arch_dir}/libonedal_parameters.a",
    ], allow_empty = True),
    deps = [
        ":headers",
    ],
)

# Packaged oneDAL releases expose DPC libraries only in dynamic form, so no
# static-DPC consumer target is declared here.

cc_library(
    name = "core_dynamic",
    srcs = glob([
        "lib/%{arch_dir}/libonedal_core.so.%{version_binary_major}",
    ]),
    deps = [
        ":headers",
        # TODO: drop TBB once vml_ipp no longer depends on it.
        "@tbb//:tbb_binary",
    ],
)

filegroup(
    name = "core_dynamic_runtime",
    srcs = glob([
        "lib/%{arch_dir}/libonedal_core.so*",
    ], allow_empty = True),
)

cc_library(
    name = "thread_dynamic",
    srcs = glob([
        "lib/%{arch_dir}/libonedal_thread.so.%{version_binary_major}",
    ]),
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
    ],
)

filegroup(
    name = "thread_dynamic_runtime",
    srcs = glob([
        "lib/%{arch_dir}/libonedal_thread.so*",
    ], allow_empty = True),
)

cc_library(
    name = "onedal_dynamic",
    srcs = glob([
        # Link through the SONAME symlinks. Bazel's _solib runfiles then expose
        # names like libonedal.so.4, matching DT_NEEDED in test executables.
        "lib/%{arch_dir}/libonedal.so.%{version_binary_major}",
    ]) + glob([
        # Separate-layout only; see `onedal_static` above.
        "lib/%{arch_dir}/libonedal_parameters.so.%{version_binary_major}",
    ], allow_empty = True),
    deps = [
        ":headers",
        "@mkl//:mkl_static",
    ],
)

filegroup(
    name = "onedal_dynamic_runtime",
    srcs = glob([
        "lib/%{arch_dir}/libonedal.so*",
        "lib/%{arch_dir}/libonedal_parameters.so*",
    ], allow_empty = True),
)

cc_library(
    name = "onedal_dynamic_dpc",
    srcs = glob([
        "lib/%{arch_dir}/libonedal_dpc.so.%{version_binary_major}",
        # Separate-layout only; see `onedal_static` above.
        "lib/%{arch_dir}/libonedal_parameters_dpc.so.%{version_binary_major}",
    ], allow_empty = True),
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
    ],
)
