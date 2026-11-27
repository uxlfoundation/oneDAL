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
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/backend:model_impl",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/detail:optimizers",
    ],
)

dal_module(
    name = "parameters",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/parameters",
    ],
)

dal_module(
    name = "logistic_regression",
    dal_deps = [
        ":core",
        ":parameters",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/detail",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/backend",
    ],
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    compile_as = ["dpc++"],
    hdrs = glob(
        [
            "test/*.hpp",
        ],
        exclude = [
            "test/mpi_*.hpp",
            "test/ccl_*.hpp",
        ],
    ),
    srcs = glob(
        [
            "test/*.cpp",
        ],
        exclude = [
            "test/mpi_*.cpp",
            "test/ccl_*.cpp",
        ],
    ),
    dal_deps = [
        ":logistic_regression",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
    ],
)
