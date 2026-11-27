load("@onedal//dev/bazel:dal.bzl", "dal_test_suite")

package(default_visibility = ["//visibility:public"])

dal_test_suite(
    name = "tests",
    compile_as = ["c++"],
    private = True,
    srcs = glob([
        "*.cpp",
    ]),
    dal_deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
    ],
)
