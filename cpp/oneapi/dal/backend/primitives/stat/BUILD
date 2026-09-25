package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "stat",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:blas",
        "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
    ],
)

dal_test_suite(
    name = "perf_tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    private = True,
    srcs = glob([
        "test/perf_*_dpc.cpp",
    ]),
    dal_deps = [
        ":stat",
    ],
    tags = [
        "exclusive",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    private = True,
    srcs = glob([
        "test/*_dpc.cpp",
    ], exclude=[
        "test/*perf*.cpp",
    ]),
    dal_deps = [
        ":stat",
    ],
    tags = [
        "exclusive",
    ],
)
