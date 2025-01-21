load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "pca",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:lapack",
        "@onedal//cpp/oneapi/dal/backend/primitives:stat",
        "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/pca:kernel",
        "@onedal//cpp/daal/src/algorithms/pca/transform:kernel",
    ],
)

dal_test_suite(
    name = "backend_tests",
    srcs = glob([
        "backend/test/*.cpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    private = True,
    framework = "catch2",
)

dal_test_suite(
    name = "interface_tests",
    srcs = glob([
        "test/*.cpp",
    ]),
    hdrs = glob([
    "test/*.hpp",
    ]),
    dal_deps = [
        ":pca",
    ],
    framework = "catch2",
)

dal_test_suite(
    name = "tests",
    tests = [
        ":backend_tests",
        ":interface_tests",
    ],
)
