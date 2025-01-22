package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:dal.bzl",
    "dal_test_suite",
    "dal_collect_test_suites",
)
load("@onedal//dev/bazel:daal.bzl",
    "daal_module",
    "daal_static_lib",
    "daal_dynamic_lib",
    "daal_algorithms",
    "daal_generate_version",
    "daal_patch_kernel_defines",
)

daal_module(
    name = "microvmlipp",
    override_deps_lib_tag = True,
    deps = select({
        "@config//:backend_ref": [ ],
        "//conditions:default": [
                                    "@mkl//:mkl_thr",
                                ],
        }),
)

daal_module(
    name = "mathbackend_thread",
    override_deps_lib_tag = True,
    deps = select({
        "@config//:backend_ref": [ "@openblas//:openblas",
                                 ],
        "//conditions:default": [ "@mkl//:mkl_thr",
                                ],
        }),
)

daal_patch_kernel_defines(
    name = "kernel_defines",
    src = "include/services/internal/daal_kernel_defines.h",
)

daal_module(
    name = "public_includes",
    hdrs = glob(["include/**/*.h"], exclude=["**/daal_kernel_defines.h"]),
    includes = [ "include" ],
)

daal_module(
    name = "includes",
    hdrs = [ ":kernel_defines",
    ],
    includes = [ "." ],
    defines = [
        "DAAL_NOTHROW_EXCEPTIONS",
        "DAAL_HIDE_DEPRECATED",
    ],
    deps = select({
        "@config//:backend_ref": [
                                   ":public_includes",
                                   "@openblas//:headers",
                                 ],
        "//conditions:default": [
                                  ":public_includes",
                                  "@mkl//:headers",
                                ],
        }),
)

daal_generate_version(
    name = "version",
    out = "daal_version_gen.h",
)

daal_module(
    name = "service_headers",
    hdrs = glob([
        "src/services/**/*.h",
        "src/externals/**/*.h",
        "src/algorithms/*.h",
    ]) + [
        ":version",
    ],
    deps = [
        ":includes",
    ],
)

daal_module(
    name = "threading_headers",
    hdrs = glob(["src/threading/**/*.h"]),
    deps = [
        ":service_headers",
    ],
)

daal_module(
    name = "services",
    srcs = glob([
            "src/services/**/*.cpp",
            "src/externals/**/*.cpp",
            "src/algorithms/*.cpp",
        ],
        exclude = ["src/externals/**/*_win_dll.cpp"],
    ),
    deps = [
        ":service_headers",
        ":threading_headers",
        ":microvmlipp",
    ],
)

daal_module(
    name = "data_management",
    hdrs = glob(["src/data_management/**/*.h", "src/data_management/**/*.i"]),
    srcs = glob(["src/data_management/**/*.cpp"]),
    deps = [
        ":services",
    ],
)

daal_module(
    name = "threading_tbb",
    srcs = glob(["src/threading/**/*.cpp"]),
    local_defines = [
        "__TBB_NO_IMPLICIT_LINKAGE",
        "__TBB_LEGACY_MODE",
        "TBB_SUPPRESS_DEPRECATED_MESSAGES",
        "TBB_USE_ASSERT=0",
    ],
    deps = select({
            "@config//:backend_ref": [
                                          ":threading_headers",
                                          ":mathbackend_thread",
                                          "@tbb//:tbb",
                                          "@tbb//:tbbmalloc",
                                        ],
            "//conditions:default": [
                                      ":threading_headers",
                                      ":mathbackend_thread",
                                      "@tbb//:tbb",
                                      "@tbb//:tbbmalloc",
                                    ],
        }),
)

daal_module(
    name = "engines",
    deps = [
        "@onedal//cpp/daal/src/algorithms/engines:kernel",
    ],
)

daal_module(
    name = "core",
    deps = [
        ":services",
        ":data_management",
    ],
)

daal_algorithms(
    name = "all_algorithms",
    algorithms = [
        "adaboost",
        "assocrules",
        "brownboost",
        "cholesky",
        "classifier",
        "cordistance",
        "cosdistance",
        "covariance",
        "dbscan",
        "decision_tree",
        "distributions",
        "dtrees",
        "dtrees/forest/classification",
        "dtrees/forest/regression",
        "dtrees/gbt/regression",
        "dtrees/gbt/classification",
        "elastic_net",
        "em",
        "engines",
        "implicit_als",
        "k_nearest_neighbors",
        "kernel_function",
        "kmeans",
        "lasso_regression",
        "linear_model",
        "linear_regression",
        "logistic_regression",
        "logitboost",
        "low_order_moments",
        "multiclassclassifier",
        "naivebayes",
        "normalization/minmax",
        "normalization/zscore",
        "objective_function",
        "objective_function/cross_entropy_loss",
        "objective_function/logistic_loss",
        "objective_function/mse",
        "optimization_solver",
        "optimization_solver/sgd",
        "optimization_solver/lbfgs",
        "optimization_solver/adagrad",
        "optimization_solver/saga",
        "optimization_solver/coordinate_descent",
        "outlierdetection_bacon",
        "outlierdetection_multivariate",
        "outlierdetection_univariate",
        "pca",
        "pca/metrics",
        "pca/transform",
        "pivoted_qr",
        "qr",
        "quantiles",
        "regression",
        "ridge_regression",
        "sorting",
        "stump",
        "svd",
        "svm",
        "weak_learner/inner",
    ],
)

daal_static_lib(
    name = "core_static",
    lib_name = "onedal_core",
    deps = [
        ":core",
        ":all_algorithms",
    ],
)


daal_static_lib(
    name = "thread_static",
    lib_name = "onedal_thread",
    deps = [
        ":threading_tbb",
    ],
)

daal_dynamic_lib(
    name = "core_dynamic",
    lib_name = "onedal_core",
    deps = [
        ":core",
        ":all_algorithms",
    ],
)

daal_dynamic_lib(
    name = "thread_dynamic",
    lib_name = "onedal_thread",
    deps = [
        ":threading_tbb",
    ],
)

daal_module(
    name = "threading_static",
    deps = select({
        "@config//:test_thread_mode_par": [
            "@onedal//cpp/daal:thread_static",
        ]
    }),
)

daal_module(
    name = "threading_dynamic",
    deps = select({
        "@config//:test_thread_mode_par": [
            "@onedal//cpp/daal:thread_dynamic",
        ]
    }),
)

daal_module(
    name = "threading_release_static",
    deps = select({
        "@config//:test_thread_mode_par": [
            "@onedal_release//:thread_static",
        ]
    }),
)

daal_module(
    name = "threading_release_dynamic",
    deps = select({
        "@config//:test_thread_mode_par": [
            "@onedal_release//:thread_dynamic",
        ]
    }),
)

filegroup(
    name = "all_static",
    srcs = [
        ":core_static",
        ":thread_static",
    ],
)

dal_test_suite(
    name = "unit_tests",
    framework = "catch2",
    srcs = glob([
        "test/*.cpp",
    ]),
)

dal_collect_test_suites(
    name = "tests",
    root = "@onedal//cpp/daal/src/algorithms",
    modules = [
        "dtrees/gbt/regression"
    ],
    tests = [
        ":unit_tests",
    ],
)
