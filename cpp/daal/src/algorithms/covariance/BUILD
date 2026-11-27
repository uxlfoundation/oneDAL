load("@onedal//dev/bazel:daal.bzl", "daal_module")
load("@onedal//dev/bazel:dal.bzl", "dal_test_suite")

package(default_visibility = ["//visibility:public"])

daal_module(
    name = "kernel",
    hdrs = glob([
        "**/*.h",
        "**/*.i",
    ]),
    srcs = glob(["*.cpp"]),
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
    ],
)

dal_test_suite(
    name = "tests",
    compile_as = ["c++"],
    private = True,
    srcs = glob([
        "test/*unit.cpp",
    ]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/covariance:kernel",
    ],
)
