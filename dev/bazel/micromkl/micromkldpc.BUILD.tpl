package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:cc.bzl", "cc_module")

cc_module(
    name = "headers",
    hdrs = glob(["include/*.h"]),
    system_includes = [ "include" ],
)

cc_module(
    name = "mkl_dpc",
    libs = [
        "libdaal_sycl.a",
    ],
    deps = [
        ":headers",
    ],
)
