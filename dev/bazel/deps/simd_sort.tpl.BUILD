package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob(["src/*.h", "src/*.hpp"]),
    includes = [ "src" ],
    defines = [ "DAAL_USE_SIMD_SORT" ],
)
