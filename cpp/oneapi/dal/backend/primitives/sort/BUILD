package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "sort",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
    dpc_deps = [
        "@dpl//:headers",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    private = True,
    srcs = glob([
        "test/*_dpc.cpp",
    ],
    exclude=[
        "test/perf_*.cpp",
    ]),
    dal_deps = [
        ":sort",
    ],
    tags = [
        "exclusive",
    ],
)

dal_test_suite(
    name = "perf_tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    private = True,
    srcs = glob([
        "test/perf_*.cpp",
    ]),
    dal_deps = [
        ":sort",
    ],
)
