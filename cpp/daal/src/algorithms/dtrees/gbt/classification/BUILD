package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:public_includes_classifier",
        "@onedal//cpp/daal:public_includes_gradient_boosted_trees",
        "@onedal//cpp/daal:core",
        "@onedal//cpp/daal:engines",
        "@onedal//cpp/daal/src/algorithms/classifier:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt:kernel",
        "@onedal//cpp/daal/src/algorithms/dtrees/gbt/regression:kernel",
        "@onedal//cpp/daal/src/algorithms/objective_function/cross_entropy_loss:kernel",
    ],
)
