load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "gpu",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal/backend/primitives:blas",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:lapack",
        "@onedal//cpp/oneapi/dal/backend/primitives:objective_function",
        "@onedal//cpp/oneapi/dal/backend/primitives:optimizers",
        "@onedal//cpp/oneapi/dal/backend/primitives:sparse_blas",
        "@onedal//cpp/oneapi/dal/algo/logistic_regression:core",
    ],
)
