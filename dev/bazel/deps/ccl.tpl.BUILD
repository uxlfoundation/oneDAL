package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    features = [ "dpc++" ],
    hdrs = glob([
        "include/oneapi/**/*.hpp",
        "include/oneapi/**/*.h",
    ]),
    # oneDAL includes `<ccl.hpp>`; the samples include `<oneapi/ccl.hpp>`.
    includes = [ "include/oneapi", "include" ],
)

cc_library(
    name = "libccl",
    srcs = [
        "lib/libccl.so.1.0",
    ],
    # libccl.so needs the DPC++ runtime; a host link does not get it from -fsycl.
    # Linux only: the Windows release build analyzes this target, and @opencl does not resolve there.
    deps = select({
        "@platforms//os:linux": ["@opencl//:opencl_binary"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "ccl",
    deps = [
        ":headers",
        ":libccl",
    ],
)
