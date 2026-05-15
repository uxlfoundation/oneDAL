package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ], allow_empty = True),
    includes = ["include"],
)

cc_library(
    name = "mkl_static",
    srcs = [
        # Match Make's Windows CPU MKL linkage: static MKL interface, core, and TBB-thread archives.
        "lib/mkl_intel_ilp64.lib",
        "lib/mkl_tbb_thread.lib",
        "lib/mkl_core.lib",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb",
    ],
    defines = ["MKL_ILP64"],
)

cc_library(
    name = "mkl_core",
    deps = [
        ":headers",
        ":mkl_static",
    ],
)

cc_library(
    name = "mkl_dpc_utils",
)

cc_library(
    name = "mkl_dpc",
    data = glob(["bin/*.dll"], allow_empty = True),
    deps = [
        ":headers",
        ":mkl_dpc_utils",
    ],
    defines = ["MKL_LP64"],
)
