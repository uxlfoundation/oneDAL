load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "backend",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/covariance/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/covariance/backend/gpu",
    ],
)
