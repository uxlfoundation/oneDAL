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
        "lib/intel64/libonedal_parameters.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_static_dpc",
    srcs = [
        "lib/intel64/libonedal_dpc.a",
        "lib/intel64/libonedal_parameters_dpc.a",
    ],
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
    ],
)

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
        "lib/intel64/libonedal_parameters.so.%{version_binary_major}",
    ]),
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
        # Link through the SONAME symlink for the same _solib/runfiles reason.
        "lib/intel64/libonedal_parameters_dpc.so.%{version_binary_major}",
    ], allow_empty=True),
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
    ],
)
