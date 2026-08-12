package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"], allow_empty = True),
    includes = ["include"],
)

# `tbb12.lib` and `tbbmalloc.lib` are *import* libraries for `tbb12.dll` and
# `tbbmalloc.dll`, not static archives. Listing them in `srcs` makes Bazel
# classify them as static libraries, which then get merged into oneDAL's own
# static libraries by `lib.exe` (dev/bazel/cc/link.bzl `_merge_static_libs`).
# Two import libraries both define `__NULL_IMPORT_DESCRIPTOR`, so the merge
# emits `LNK4006: __NULL_IMPORT_DESCRIPTOR already defined in tbb12.lib`, and
# the released `onedal_core.lib` ends up carrying TBB import descriptors that
# the Make package does not ship. `cc_import` with `system_provided = True`
# keeps them on the link line only; the DLLs are resolved from `PATH` at run
# time, exactly as with the Make build.
cc_import(
    name = "tbb_import",
    interface_library = "lib/tbb12.lib",
    system_provided = True,
)

cc_library(
    name = "tbb",
    data = ["bin/tbb12.dll"],
    deps = [
        ":headers",
        ":tbb_import",
    ],
)

filegroup(
    name = "tbb_runtime",
    srcs = [
        "bin/tbb12.dll",
        "bin/tbbmalloc.dll",
    ],
)

cc_library(
    name = "tbb_binary",
    deps = [":tbb"],
)

cc_import(
    name = "tbbmalloc_import",
    interface_library = "lib/tbbmalloc.lib",
    system_provided = True,
)

cc_library(
    name = "tbbmalloc",
    data = ["bin/tbbmalloc.dll"],
    deps = [
        ":headers",
        ":tbbmalloc_import",
    ],
)

cc_library(
    name = "tbbmalloc_binary",
    deps = [":tbbmalloc"],
)
