package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_toolchain")
cc_library(
    name = "headers",
    hdrs = glob(["include/**/**/*"]),
    includes = [ "include" ],
)
