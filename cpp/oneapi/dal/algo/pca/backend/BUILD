package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "backend",
    dal_deps = [
        "@onedal//cpp/oneapi/dal/algo/pca/backend/cpu",
        "@onedal//cpp/oneapi/dal/algo/pca/backend/gpu",
    ],
)

dal_module(
    name = "common",
    hdrs = glob(["*.hpp"]) + [
        "@onedal//cpp/oneapi/dal:cpu_dispatcher",
    ],
    dal_deps = [
        "@onedal//cpp/oneapi/dal:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
    ],
    srcs = [
        "sign_flip_cpu.cpp",
    ],
    extra_deps = [
        "@onedal//cpp/oneapi:include_root",
    ],
)
