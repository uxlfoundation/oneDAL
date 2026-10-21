package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
                "@onedal//cpp/daal/src/algorithms/dtrees:kernel",
        # the tree engine the training kernels run on (df_forest_trainer.i)
        "@onedal//cpp/daal/src/algorithms/dtrees/engine:kernel",
        "@onedal//cpp/daal/src/algorithms/distributions:kernel",
    ],
)
