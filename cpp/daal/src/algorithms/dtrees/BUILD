package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl", "daal_module")

daal_module(
    name = "kernel",
    auto = True,
    deps = [
        "@onedal//cpp/daal:public_includes_classifier",
        "@onedal//cpp/daal:public_includes_decision_forest",
        "@onedal//cpp/daal:public_includes_engines",
        "@onedal//cpp/daal:public_includes_gradient_boosted_trees",
        "@onedal//cpp/daal:public_includes_regression",
        "@onedal//cpp/daal:public_includes_tree_utils",
        "@onedal//cpp/daal:core",
    ],
)
