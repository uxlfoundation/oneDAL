package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

dal_module(
    name = "sparse_blas",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/backend/primitives:common",
        "@onedal//cpp/oneapi/dal/backend/primitives:blas",
    ],
)

# Deselected under `--dpc_math_backend=onemath`, where the whole domain throws
# `unimplemented`: oneMath ships the newer sparse specification (`init_csr_matrix`
# plus descriptor-driven `spmv`/`spmm`) rather than the handle API these
# primitives are written against. `@platforms//:incompatible` makes Bazel skip
# the targets instead of running them to a certain failure.
dal_test_suite(
    name = "tests",
    framework = "catch2",
    compile_as = [ "dpc++" ],
    private = True,
    target_compatible_with = select({
        "@config//:dpc_math_backend_onemath": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    hdrs = glob([
        "test/*.hpp",
    ]),
    srcs = glob([
        "test/*_dpc.cpp",
    ], exclude=[
        "test/*perf*.cpp",
    ]),
    dal_deps = [
        ":sparse_blas",
    ],
)
