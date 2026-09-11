package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = [ "include" ],
)

cc_library(
    name = "openrng",
    srcs = [
        "lib/libopenrng.a",
    ],
    deps = [
        ":headers",
    ],
)
