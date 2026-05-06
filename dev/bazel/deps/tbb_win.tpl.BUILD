package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"], allow_empty = True),
    includes = ["include"],
)

cc_library(
    name = "tbb",
    srcs = ["lib/tbb12.lib"],
    data = ["bin/tbb12.dll"],
    deps = [":headers"],
)

cc_library(
    name = "tbbmalloc",
    srcs = ["lib/tbbmalloc.lib"],
    data = ["bin/tbbmalloc.dll"],
    deps = [":headers"],
)
