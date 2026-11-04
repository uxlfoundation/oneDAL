package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:cc.bzl", "cc_module")

# `system_includes` passes the headers via `-isystem`, as Make does, so that
# oneDPL's own warnings do not fail the `-Werror` build.
cc_module(
    name = "headers",
    hdrs = glob(["include/**/**/*"]),
    system_includes = [ "include" ],
)
