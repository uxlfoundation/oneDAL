package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "sparse_blas",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:blas",
    ],
)

dal_test_suite(
    name = "perf_tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    private = True,
    hdrs = glob([
        "test/*.hpp",
    ]),
    srcs = glob([
        "test/*perf_dpc.cpp",
    ]),
    dal_deps = [
        ":sparse_blas",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    private = True,
    hdrs = glob([
        "test/*.hpp",
    ]),
    srcs = glob([
        "test/*_dpc.cpp",
    ], exclude=[
        "test/*perf*.cpp",
    ]),
    dal_deps = [
        ":sparse_blas",
    ],
)
