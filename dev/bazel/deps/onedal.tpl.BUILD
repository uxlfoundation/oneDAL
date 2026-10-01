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
        "lib/intel64/libonedal_core.a",
    ],
    deps = [
        ":headers",
        # TODO: Currently vml_ipp lib depends on TBB, but it shouldn't
        #       Remove TBB from deps once problem with vml_ipp is resolved
        "@tbb//:tbb_binary",
        "@mkl//:mkl_static",
    ],
)

cc_library(
    name = "thread_static",
    srcs = [
        "lib/intel64/libonedal_thread.a",
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
        "lib/intel64/libonedal.a",
    ] + glob([
        # Present only when the package ships separate parameter libraries.
        # `repos.bzl` symlinks these from `optional_libs` after reading the
        # layout out of the package's own oneDALConfig.cmake, so an absent
        # entry here means a folded package, not a broken one.
        "lib/intel64/libonedal_parameters.a",
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
        "lib/intel64/libonedal_core.so.%{version_binary_major}",
    ]),
    deps = [
        ":headers",
        # TODO: Currently vml_ipp lib depends on TBB, but it shouldn't
        #       Remove TBB from deps once problem with vml_ipp is resolved
        "@tbb//:tbb_binary",
    ],
)

filegroup(
    name = "core_dynamic_runtime",
    srcs = glob([
        "lib/intel64/libonedal_core.so*",
    ], allow_empty = True),
)

cc_library(
    name = "thread_dynamic",
    srcs = glob([
        "lib/intel64/libonedal_thread.so.%{version_binary_major}",
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
        "lib/intel64/libonedal_thread.so*",
    ], allow_empty = True),
)

cc_library(
    name = "onedal_dynamic",
    srcs = glob([
        # Link through the SONAME symlinks. Bazel's _solib runfiles then expose
        # names like libonedal.so.4, matching DT_NEEDED in test executables.
        "lib/intel64/libonedal.so.%{version_binary_major}",
    ]) + glob([
        # Separate-layout only; see `onedal_static` above.
        "lib/intel64/libonedal_parameters.so.%{version_binary_major}",
    ], allow_empty = True),
    deps = [
        ":headers",
        "@mkl//:mkl_static",
    ],
)

filegroup(
    name = "onedal_dynamic_runtime",
    srcs = glob([
        "lib/intel64/libonedal.so*",
        "lib/intel64/libonedal_parameters.so*",
    ], allow_empty = True),
)

cc_library(
    name = "onedal_dynamic_dpc",
    srcs = glob([
        "lib/intel64/libonedal_dpc.so.%{version_binary_major}",
        # Separate-layout only; see `onedal_static` above.
        "lib/intel64/libonedal_parameters_dpc.so.%{version_binary_major}",
    ], allow_empty = True),
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
    ],
)
