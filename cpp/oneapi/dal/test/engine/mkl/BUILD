load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_test_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_test_module(
    name = "mkl",
    auto = True,
    dal_test_deps = [
        "@onedal//cpp/oneapi:include_root",
    ],
    extra_deps = [{
        "@config//:backend_ref": [
            "@openblas//:openblas",
        ],
        "//conditions:default": [
            "@mkl//:mkl_core",
        ],
    }],
)
