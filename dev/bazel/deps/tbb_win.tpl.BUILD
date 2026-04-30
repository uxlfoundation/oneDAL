package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"], allow_empty = True),
    includes = ["include"],
)

cc_import(
    name = "tbb_import",
    interface_library = "lib/tbb12.lib",
    shared_library = "bin/tbb12.dll",
)

cc_library(
    name = "tbb",
    data = glob(["bin/*.dll"], allow_empty = True),
    deps = [
        ":headers",
        ":tbb_import",
    ],
)

cc_import(
    name = "tbbmalloc_import",
    interface_library = "lib/tbbmalloc.lib",
    shared_library = "bin/tbbmalloc.dll",
)

cc_library(
    name = "tbbmalloc",
    data = glob(["bin/*.dll"], allow_empty = True),
    deps = [":tbbmalloc_import"],
)
