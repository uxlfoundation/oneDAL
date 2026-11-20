load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "backend",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/decision_forest/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/decision_forest/backend/gpu",
    ],
)

dal_module(
    name = "model_impl",
    hdrs = glob(["*.hpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo:decision_tree",
    ],
)
