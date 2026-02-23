package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "opencl_binary",
    srcs = glob(
        [
            "*.so",
            "*.so.*",
        ],
        exclude = ["*.py", "*.cmake", "*.a"],
    ),
    linkopts = ["-lOpenCL"],
    visibility = ["//visibility:public"],
)
