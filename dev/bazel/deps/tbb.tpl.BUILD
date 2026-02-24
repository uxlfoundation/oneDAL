package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = [ "include" ],
)

cc_library(
    name = "tbb_binary",
    srcs = [
        "lib/libtbb.so.12",
    ],
    linkopts = [
        "-lpthread",
    ],
)

cc_library(
    name = "tbbmalloc_binary",
    srcs = [
        "lib/libtbbmalloc.so.2",
    ],
)

cc_library(
    name = "tbb",
    deps = [
        ":headers",
        ":tbb_binary",
    ],
)

cc_library(
    name = "tbbmalloc",
    deps = [
        ":headers",
        ":tbbmalloc_binary",
    ],
)
