package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "search",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:blas",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:distance",
        "@onedal//cpp/oneapi/dal/backend/primitives:selection",
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
        ":search",
    ],
)
