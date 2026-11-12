package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")

# The link resolves `-lgomp` here instead of in the compiler's own libraries.
cc_library(
    name = "openmp_binary",
    srcs = [
        "lib/libgomp.so",
    ],
)
