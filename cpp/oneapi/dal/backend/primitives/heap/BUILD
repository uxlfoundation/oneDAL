load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "heap",
    auto = True,
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = ["c++"],
    private = True,
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        ":heap",
    ],
)
