package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/oneapi/**/*.hpp",
    ]),
    includes = ["include"],
)

# ===== STATIC LIBRARIES =====

cc_library(
    name = "core_static",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/onedal_core.lib"],
        "//conditions:default": ["lib/intel64/libonedal_core.a"],
    }),
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
    ],
)

cc_library(
    name = "thread_static",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/onedal_thread.lib"],
        "//conditions:default": ["lib/intel64/libonedal_thread.a"],
    }),
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
    ],
)

cc_library(
    name = "parameters_static",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/onedal_parameters.lib"],
        "//conditions:default": ["lib/intel64/libonedal_parameters.a"],
    }),
    deps = [":headers"],
)

cc_library(
    name = "onedal_static",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal.lib"],
        "//conditions:default": ["lib/intel64/libonedal.a"],
    }),
    deps = [
        ":headers",
        ":parameters_static",
    ],
)

cc_library(
    name = "parameters_static_dpc",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal_parameters_dpc.lib"],
        "//conditions:default": ["lib/intel64/libonedal_parameters_dpc.a"],
    }),
    deps = [":headers"],
)

cc_library(
    name = "onedal_static_dpc",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal_dpc.lib"],
        "//conditions:default": ["lib/intel64/libonedal_dpc.a"],
    }),
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
        ":parameters_static_dpc",
    ],
)

# ===== DYNAMIC LIBRARIES =====

cc_library(
    name = "core_dynamic",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal_core.lib"],
        "//conditions:default": ["lib/intel64/libonedal_core.so"],
    }),
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
    ],
)

cc_library(
    name = "thread_dynamic",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal_thread.lib"],
        "//conditions:default": ["lib/intel64/libonedal_thread.so"],
    }),
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
    ],
)

cc_library(
    name = "parameters_dynamic",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal_parameters.lib"],
        "//conditions:default": ["lib/intel64/libonedal_parameters.so"],
    }),
    deps = [":headers"],
)

cc_library(
    name = "onedal_dynamic",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal.lib"],
        "//conditions:default": ["lib/intel64/libonedal.so"],
    }),
    deps = [
        ":headers",
        ":parameters_dynamic",
    ],
)

cc_library(
    name = "parameters_dynamic_dpc",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal_parameters_dpc.lib"],
        "//conditions:default": ["lib/intel64/libonedal_parameters_dpc.so"],
    }),
    deps = [":headers"],
)

cc_library(
    name = "onedal_dynamic_dpc",
    srcs = select({
        "@platforms//os:windows": ["lib/intel64/libonedal_dpc.lib"],
        "//conditions:default": ["lib/intel64/libonedal_dpc.so"],
    }),
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
        ":parameters_dynamic_dpc",
    ],
)
