package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
filegroup(
    name = "mpi_runfiles",
    srcs = glob([
        "bin/**/*",
    ]),
)

sh_binary(
    name = "mpiexec",
    srcs = [
        "bin/mpiexec",
    ],
    data = [
        ":mpi_runfiles",
    ]
)

cc_library(
    name = "headers",
    hdrs = glob(["include/**/*.h"]),
    includes = [ "include" ],
)

cc_library(
    name = "libmpi",
    srcs = [
        "lib/release/libmpi.so.12",
    ],
)

cc_library(
    name = "libfabric",
    srcs = [
        "libfabric/lib/libfabric.so.1",
    ],
)

filegroup(
    name = "fi",
    srcs = glob([
        "libfabric/lib/prov/*.so",
    ]),
)

cc_library(
    name = "mpi",
    deps = [
        ":headers",
        ":libmpi",
        ":libfabric",
    ],
)
