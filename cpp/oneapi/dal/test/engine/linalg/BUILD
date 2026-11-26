load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_test_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_test_module(
    name = "linalg",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/table",
    ],
    dal_test_deps = [
        "@onedal//cpp/oneapi/dal/test/engine/mkl",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = ["c++"],
    srcs = glob(["test/*.cpp"]),
    dal_deps = [
        ":linalg",
    ],
)
