load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "selection",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:heap",
        "@onedal//cpp/oneapi/dal/backend/primitives:rng",
    ],
)

dal_test_suite(
    name = "tests",
    private = True,
    framework = "catch2",
    compile_as = ["dpc++"],
    srcs = glob(
        [
            "test/*_dpc.cpp",
        ],
        exclude = [
            "test/*perf*.cpp",
        ],
    ),
    dal_deps = [
        ":selection",
    ],
    tags = [
        "exclusive",
    ],
)

dal_test_suite(
    name = "perf_tests",
    framework = "catch2",
    compile_as = ["dpc++"],
    private = True,
    srcs = glob([
        "test/perf_*.cpp",
    ]),
    dal_deps = [
        ":selection",
    ],
)
