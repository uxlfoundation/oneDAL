package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "triangle_counting",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:intersection",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [],
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ],
    exclude = [
        "test/*_gpu_test.cpp",
    ]),
    dal_deps = [
        ":triangle_counting",
    ],
)

dal_test_suite(
    name = "gpu_tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    srcs = glob([
        "test/*_gpu_test.cpp",
    ]),
    dal_deps = [
        ":triangle_counting",
    ],
)
