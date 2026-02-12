package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_toolchain")
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
            "lib/libgfortran.a", 
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
