package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = [
        "include",
    ],
    defines = [
        "MKL_ILP64"
    ],
)

cc_library(
    name = "mkl_static",
    srcs = [
        "lib/libmkl_core.a",
        "lib/libmkl_intel_ilp64.a",
        "lib/libmkl_tbb_thread.a",
    ],
    linkopts = [
        # The source libraries have circular symbol dependencies. To successfully build this cc_library,
        # oneMKL requires wrapping the libraries with -Wl,--start-group and -Wl,--end-group.
        "-Wl,--start-group",
        "$(location lib/libmkl_core.a)",
        "$(location lib/libmkl_intel_ilp64.a)",
        "$(location lib/libmkl_tbb_thread.a)",
        "-Wl,--end-group",
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    deps = [
        ":headers",
        "@opencl//:opencl_binary",
    ],
    alwayslink = 1,
    linkstatic = 1,
)

cc_library(
    name = "mkl_core",
    linkopts = [
        "-lpthread",
    ],
    deps = [
        ":headers",
        ":mkl_static",
    ]
)

cc_library(
    name = "mkl_dpc_utils",
    linkopts = [
        "-fsycl-max-parallel-link-jobs=16",
    ],
    srcs = [
        "lib/libmkl_sycl_blas.so",
        "lib/libmkl_sycl_lapack.so",
        "lib/libmkl_sycl_sparse.so",
        "lib/libmkl_sycl_dft.so",
        "lib/libmkl_sycl_vm.so",
        "lib/libmkl_sycl_rng.so",
        "lib/libmkl_sycl_stats.so",
        "lib/libmkl_sycl_data_fitting.so",
        "lib/libmkl_sycl_blas.so.6",
        "lib/libmkl_sycl_lapack.so.6",
        "lib/libmkl_sycl_sparse.so.6",
        "lib/libmkl_sycl_dft.so.6",
        "lib/libmkl_sycl_vm.so.6",
        "lib/libmkl_sycl_rng.so.6",
        "lib/libmkl_sycl_stats.so.6",
        "lib/libmkl_sycl_data_fitting.so.6",
    ],
)

cc_library(
    name = "mkl_dpc",
    # TODO: add a mechanism to get attr from bazel command(it's not available for now)
    linkopts = [
        # Currently its hardcoded to 16 to get the best trade-off between linking speedup and resources used.
        # If the number of processors on machine is below 16 it will be defaulted to `nproc`.
        "-fsycl-max-parallel-link-jobs=16",
    ],
    srcs = [
        "lib/libmkl_sycl.so",
    ],
    deps = [
        ":headers",
        ":mkl_dpc_utils",
    ],
)
