package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/oneapi/**/*.hpp",
    ]),
    includes = ["include"],
)

cc_library(
    name = "core_static",
    srcs = [
        "lib/intel64/onedal_core.lib",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@mkl//:mkl_static",
    ],
)

cc_library(
    name = "thread_static",
    srcs = [
        "lib/intel64/onedal_thread.lib",
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
        "lib/intel64/onedal.lib",
        "lib/intel64/onedal_parameters.lib",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_static_dpc",
    srcs = [],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "core_dynamic",
    srcs = [
        "lib/intel64/onedal_core_dll.lib",
    ],
    data = [
        "redist/intel64/onedal_core.%{version_binary_major}.dll",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
    ],
)

cc_library(
    name = "thread_dynamic",
    srcs = [],
    data = [
        "redist/intel64/onedal_thread.%{version_binary_major}.dll",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
    ],
)

cc_library(
    name = "onedal_dynamic",
    srcs = [
        "lib/intel64/onedal_dll.lib",
        "lib/intel64/onedal_parameters_dll.lib",
    ],
    data = [
        "lib/intel64/onedal.%{version_binary_major}.dll",
        "lib/intel64/onedal_parameters.%{version_binary_major}.dll",
    ],
    deps = [
        ":headers",
        "@mkl//:mkl_static",
    ],
)

cc_library(
    name = "onedal_dynamic_dpc",
    srcs = [],
    deps = [
        ":headers",
    ],
)
