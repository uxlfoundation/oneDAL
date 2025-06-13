load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "core",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/algo/pca/backend:common",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/pca:kernel",
        "@onedal//cpp/daal/src/algorithms/pca/transform:kernel",
    ],
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/pca/parameters",
        "@onedal//cpp/oneapi/dal/detail/parameters",
    ],
)

dal_module(
    name = "pca",
    dal_deps = [
        ":core",
        ":parameters",
        "@onedal//cpp/oneapi/dal/algo/pca/detail",
        "@onedal//cpp/oneapi/dal/algo/pca/backend",
    ]
)

dal_test_suite(
    name = "backend_tests",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/pca/backend/test",
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
