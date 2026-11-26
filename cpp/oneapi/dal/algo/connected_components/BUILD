load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "connected_components",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:rng",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":connected_components",
    ],
)
