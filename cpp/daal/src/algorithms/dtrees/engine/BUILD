package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")
load("@onedal//dev/bazel:dal.bzl", "dal_test_suite")

daal_module(
    name = "kernel",
    auto = False,
    hdrs = glob(["*.h", "*.i"], allow_empty = True),
    srcs = glob(["*.cpp"], allow_empty = True),
    deps = [
        "@onedal//cpp/daal:core",
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
        ":kernel",
        "@onedal//cpp/daal:engines",
    ],
)

# Binning time against the legacy binner, and the sample's rank error on Higgs. Hidden from the
# default test run; select it with --test_arg="[.benchmark]".
dal_test_suite(
    name = "binning_benchmark",
    compile_as = [ "c++" ],
    private = True,
    srcs = [
        "test/binning_benchmark.cpp",
    ],
    dal_deps = [
        ":kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
    ],
)
