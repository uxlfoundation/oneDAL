load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "detail",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/logistic_regression:core",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/backend",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/parameters",
        "@onedal//cpp/oneapi/dal/algo/newton_cg:newton_cg",
    ],
)

dal_module(
    name = "optimizers",
    hdrs = glob(["optimizer.hpp"]),
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression/backend:optimizer_impl",
        "@onedal//cpp/oneapi/dal/algo/newton_cg",
    ],
)
