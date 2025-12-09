package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/oneapi/**/*.hpp",
    ]),
    includes = [ "include" ],
)

cc_library(
    name = "core_static",
    srcs = [
        "lib/intel64/libonedal_core.a",
    ],
    deps = [
        ":headers",
        # TODO: Currently vml_ipp lib depends on TBB, but it shouldn't
        #       Remove TBB from deps once problem with vml_ipp is resolved
        "@tbb//:tbb_binary",
    ],
)

cc_library(
    name = "thread_static",
    srcs = [
        "lib/intel64/libonedal_thread.a",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
    ],
)

cc_library(
    name = "parameters_static",
    srcs = [
        "lib/intel64/libonedal_parameters.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_static",
    srcs = [
        "lib/intel64/libonedal.a",
    ],
    deps = [
        ":headers",
        ":parameters_static",
    ],
)

cc_library(
    name = "parameters_static_dpc",
    srcs = [
        "lib/intel64/libonedal_parameters_dpc.a",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_static_dpc",
    srcs = [
        "lib/intel64/libonedal_dpc.a",
    ],
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
        ":parameters_static_dpc",
    ],
)

cc_library(
    name = "core_dynamic",
    srcs = [
        "lib/intel64/libonedal_core.so",
    ],
    deps = [
        ":headers",
        # TODO: Currently vml_ipp lib depends on TBB, but it shouldn't
        #       Remove TBB from deps once problem with vml_ipp is resolved
        "@tbb//:tbb_binary",
    ],
)

cc_library(
    name = "thread_dynamic",
    srcs = [
        "lib/intel64/libonedal_thread.so",
    ],
    deps = [
        ":headers",
        "@tbb//:tbb_binary",
        "@tbb//:tbbmalloc_binary",
    ],
)

cc_library(
    name = "parameters_dynamic",
    srcs = [
        "lib/intel64/libonedal_parameters.so",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_dynamic",
    srcs = [
        "lib/intel64/libonedal.so",
    ],
    deps = [
        ":headers",
        ":parameters_dynamic",
    ],
)

cc_library(
    name = "parameters_dynamic_dpc",
    srcs = [
        "lib/intel64/libonedal_parameters_dpc.so",
    ],
    deps = [
        ":headers",
    ],
)

cc_library(
    name = "onedal_dynamic_dpc",
    srcs = [
        "lib/intel64/libonedal_dpc.so",
    ],
    deps = [
        ":headers",
        "@mkl//:mkl_dpc",
        ":parameters_dynamic_dpc",
    ],
)
