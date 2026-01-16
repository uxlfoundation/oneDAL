package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")
load("@onedal//dev/bazel:dal.bzl", "dal_test_suite")

daal_module(
    name = "kernel",
    auto = False,
    hdrs = glob(["**/*.h", "**/*.i"]),
    srcs = glob(["*.cpp"]),
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt:kernel",
        "@onedal//cpp/daal/src/algorithms/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/classifier:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
    ],
)

dal_test_suite(
    name = "tests",
    compile_as = [ "c++" ],
    private = True,
    srcs = glob([
        "test/*unit.cpp",
    ]),
    dal_deps = [
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt/regression:kernel"
    ]
)
