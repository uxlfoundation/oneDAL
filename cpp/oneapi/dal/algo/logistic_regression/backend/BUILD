load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "backend",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/backend/gpu",
    ],
)

dal_module(
    name = "model_impl",
    hdrs = glob(["model_impl.hpp"]),
)

dal_module(
    name = "optimizer_impl",
    hdrs = glob(["optimizer_impl.hpp"]),
)
