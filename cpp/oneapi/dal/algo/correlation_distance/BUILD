package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "correlation_distance",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
	    "@onedal//cpp/oneapi/dal/backend/primitives:blas",
	    "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
	    "@onedal//cpp/oneapi/dal/backend/primitives:distance",
    ],
    extra_deps = [
	    "@onedal//cpp/daal/src/algorithms/cordistance:kernel",
	    "@onedal//cpp/daal:data_management",
    ]
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":correlation_distance",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
    ],
)
