package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = [ "include" ],
)

cc_library(
    name = "openblas_core",
    #srcs = glob(["lib/libopenblas*"]),
    srcs = [
            "lib/libopenblas.a",
           ],
    linkopts = [
        "-lpthread",
#        "-lgfortran",
    ],
)

cc_library(
    name = "openblas",
    deps = [
        ":headers",
        ":openblas_core",
    ],
)

# The Linux ref build links the static `libopenblas.a`, so nothing has to travel
# with the binaries at run time. The target exists for parity with
# `openblas_win.tpl.BUILD`, where OpenBLAS is a DLL, so `_test_runtime_data()`
# in dev/bazel/dal.bzl can name it unconditionally.
filegroup(
    name = "openblas_runtime",
    srcs = [],
)
