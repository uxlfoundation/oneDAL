package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/oneapi/**/*.hpp",
    ]),
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
        "lib/intel64/libonedal_core.so.%{version_binary_major}.%{version_binary_minor}",
    ]),
    deps = [
        ":headers",
        # TODO: Currently vml_ipp lib depends on TBB, but it shouldn't
        #       Remove TBB from deps once problem with vml_ipp is resolved
        "@tbb//:tbb_binary",
    ],
)

cc_library(
    name = "thread_dynamic",
    srcs = glob([
        "lib/intel64/libonedal_thread.so.%{version_binary_major}.%{version_binary_minor}",
    ]),
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
    ],
)

cc_library(
    name = "onedal_dynamic",
    srcs = glob([
        # Use exact versioned filenames to avoid linking the same .so multiple
        # times when the release tree contains .so, .so.<major>, and
        # .so.<major>.<minor> all pointing to the same library.
        "lib/intel64/libonedal.so.%{version_binary_major}.%{version_binary_minor}",
        "lib/intel64/libonedal_parameters.so.%{version_binary_major}.%{version_binary_minor}",
    ]),
    deps = [
        ":headers",
        "@mkl//:mkl_static",
    ],
)

cc_library(
    name = "onedal_dynamic_dpc",
    srcs = glob([
        "lib/intel64/libonedal_dpc.so.%{version_binary_major}.%{version_binary_minor}",
        # Use the exact fully-versioned filename to avoid duplicate link inputs
        # from the .so/.so.<major> symlink chain in the release tree.
        "lib/intel64/libonedal_parameters_dpc.so.%{version_binary_major}.%{version_binary_minor}",
    ], allow_empty=True),
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
    ],
)
