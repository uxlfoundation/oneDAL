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
        "@onedal//cpp/oneapi/dal/algo/pca:core",
        "@onedal//cpp/oneapi/dal/algo/pca/backend",
        "@onedal//cpp/oneapi/dal/algo/pca/parameters",
    ],
)
