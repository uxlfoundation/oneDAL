load(
    "@onedal//dev/bazel:dal.bzl",
    "dal_module",
    "dal_test_suite",
)

package(default_visibility = ["//visibility:public"])

dal_module(
    name = "svm",
    auto = True,
    dal_deps = [
        "@onedal//cpp/oneapi/dal:core",
        "@onedal//cpp/oneapi/dal/algo:linear_kernel",
        "@onedal//cpp/oneapi/dal/algo:polynomial_kernel",
        "@onedal//cpp/oneapi/dal/algo:rbf_kernel",
        "@onedal//cpp/oneapi/dal/algo:sigmoid_kernel",
        "@onedal//cpp/oneapi/dal/backend/primitives:blas",
        "@onedal//cpp/oneapi/dal/backend/primitives:reduction",
        "@onedal//cpp/oneapi/dal/backend/primitives:sort",
        "@onedal//cpp/oneapi/dal/backend/primitives:selection",
    ],
    extra_deps = [
        "@onedal//cpp/daal/src/algorithms/svm:kernel",
        "@onedal//cpp/daal/src/algorithms/multiclassclassifier:kernel",
    ],
)

dal_test_suite(
    name = "interface_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
    dal_deps = [
        ":svm",
    ],
)

dal_test_suite(
    name = "gpu_tests",
    framework = "catch2",
    private = True,
    compile_as = ["dpc++"],
    srcs = glob([
        "backend/gpu/test/*.cpp",
    ]),
    dal_deps = [
        ":svm",
    ],
)

dal_test_suite(
    name = "tests",
    tests = [
        ":interface_tests",
        ":gpu_tests",
    ],
)
