package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "knn",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/algo:minkowski_distance",
        "@onedal//cpp/oneapi/dal/algo:chebyshev_distance",
        "@onedal//cpp/oneapi/dal/algo:cosine_distance",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:distance",
        "@onedal//cpp/oneapi/dal/backend/primitives:regression",
        "@onedal//cpp/oneapi/dal/backend/primitives:search",
        "@onedal//cpp/oneapi/dal/backend/primitives:selection",
        "@onedal//cpp/oneapi/dal/backend/primitives:voting",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/k_nearest_neighbors:kernel",
    ],
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    hdrs = glob([
        "test/*.hpp",
    ]),
    srcs = glob([
        "test/badarg.cpp",
        "test/batch.cpp",
        "test/serialization.cpp",
        "test/spmd.cpp",
    ]),
    dal_deps = [
        ":knn",
    ],
)

dal_test_suite(
    name = "internal_tests",
    framework = "catch2",
    hdrs = glob([
        "test/*.hpp",
    ]),
    srcs = glob([
        "test/internal.cpp",
    ]),
    dal_deps = [
        ":knn",
    ],
    private = True,
)


dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
        ":internal_tests",
    ],
)
