package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl", "dal_test_suite")

# The training kernels on the tree engine (df_forest_trainer.i) and against the legacy kernels.
dal_test_suite(
    name = "tests",
    compile_as = [ "c++" ],
    private = True,
    srcs = glob([
        "*unit.cpp",
    ]),
    dal_deps = [
        "@onedal//cpp/daal:engines",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/classification:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/forest/regression:kernel",
    ],
)
