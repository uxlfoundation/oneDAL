load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "placement",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
    ],
)

dal_test_suite(
    name = "perf_tests",
    framework = "catch2",
    compile_as = ["dpc++"],
    private = True,
    srcs = glob([
        "test/*perf_dpc.cpp",
    ]),
    dal_deps = [
        ":placement",
    ],
)

dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = ["dpc++"],
    private = True,
    srcs = glob(
        [
            "test/*_dpc.cpp",
        ],
        exclude = [
            "test/*perf*.cpp",
        ],
    ),
    dal_deps = [
        ":placement",
    ],
)
