package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ], allow_empty = True),
    includes = ["include"],
)

cc_import(
    name = "mkl_rt_import",
    interface_library = "lib/mkl_rt.lib",
    shared_library = "bin/mkl_rt.2.dll",
)

cc_library(
    name = "mkl_static",
    data = glob(["bin/*.dll"], allow_empty = True),
    deps = [
        ":headers",
        ":mkl_rt_import",
    ],
    defines = ["MKL_ILP64"],
    alwayslink = 1,
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
    deps = [":mkl_rt_import"],
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
