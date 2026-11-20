load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "backend",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/linear_regression/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/linear_regression/backend/gpu",
    ],
)

dal_module(
    name = "model_impl",
    hdrs = glob(["*.hpp"]),
)
