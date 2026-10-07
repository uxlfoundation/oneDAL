load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "table",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/util",
        "@onedal//cpp/oneapi/dal:common",
    ],
)

dal_test_suite(
    name = "private_tests",
    hdrs = [
        "backend/test/copy_convert_fixture.hpp",
    ],
    srcs = [
        # TODO: Move conversion tests to backend folder
        "test/convert.cpp",
        "test/table_builder.cpp",
        "test/table_adapter.cpp",
        "detail/test/csr_utils.cpp",
        "detail/test/homogen_utils.cpp",
        "detail/test/metadata_utils.cpp",
        "backend/test/copy_convert.cpp",
    ],
    dal_deps = [":table"],
    framework = "catch2",
    private = True,
)

dal_test_suite(
    name = "catch2_tests",
    srcs = [
        "test/row_accessor.cpp",
        "test/heterogen.cpp",
        "test/homogen.cpp",
        "test/common_serialization.cpp",
        "test/heterogen_badargs.cpp",
        "test/heterogen_serialization.cpp",
        "test/homogen_serialization.cpp",
        "test/csr_accessor.cpp",
        "test/csr_badargs.cpp",
        "test/csr.cpp",
        "test/csr_serialization.cpp",
    ],
    dal_deps = [":table"],
    framework = "catch2",
)

dal_test_suite(
    name = "gtest_tests",
    srcs = [
        "test/column_accessor.cpp",
        "test/common.cpp",
    ],
    dal_deps = [":table"],
    framework = "catch2",
)

dal_test_suite(
    name = "tests",
    tests = [
        ":private_tests",
        ":catch2_tests",
    ],
)
