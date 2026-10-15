package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob(["include/**/**/*"]),
    includes = [ "include" ],
    # conda installs the SYCL headers next to these, so they must not be -I.
    features = ["system_include_paths"],
)
