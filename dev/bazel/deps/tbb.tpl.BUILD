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

filegroup(
    name = "tbb_runtime",
    srcs = glob([
        "lib/libtbb.so*",
        "lib/libtbbmalloc.so*",
    ], allow_empty = True),
)

# Windows-only counterpart of `tbb_runtime`: there the released libraries link
# against import libraries that the package carries next to the DLLs
# (`makefile:278` stages them into `tbb/latest/lib/vc_mt`). On Linux the `.so` is
# both the link and the runtime artifact and is already in `tbb_runtime`, so
# this group is empty and the release skips it.
filegroup(
    name = "tbb_import_libs",
    srcs = [],
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
