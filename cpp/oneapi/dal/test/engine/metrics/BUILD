load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_test_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_test_module(
    name = "metrics",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/table",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = ["c++"],
    srcs = glob(["test/*.cpp"]),
    dal_deps = [
        ":metrics",
    ],
)
