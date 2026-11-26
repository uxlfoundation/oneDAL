load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "lapack",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:blas",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = ["dpc++"],
    private = True,
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":lapack",
    ],
)
