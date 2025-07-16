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
    name = "mkl_core",
    srcs = [
        "lib/libmkl_core.so.2",
        "lib/libmkl_intel_ilp64.so.2",
        "lib/libmkl_tbb_thread.so.2",
        "lib/libmkl_vml_avx512.so.2",
        "lib/libmkl_vml_def.so.2",
        "lib/libmkl_avx512.so.2",
    ],
    deps = [
        ":headers",
        "@opencl//:opencl_binary",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mkl_thr",
    linkopts = [
        "-lpthread",
    ],
    deps = [
        ":headers",
        ":mkl_core",
    ]
)

cc_library(
    name = "mkl_dpc_utils",
    srcs = [
        "lib/libmkl_sycl_blas.so",
        "lib/libmkl_sycl_lapack.so",
        "lib/libmkl_sycl_sparse.so",
        "lib/libmkl_sycl_dft.so",
        "lib/libmkl_sycl_vm.so",
        "lib/libmkl_sycl_rng.so",
        "lib/libmkl_sycl_stats.so",
        "lib/libmkl_sycl_data_fitting.so",
        "lib/libmkl_sycl_blas.so.5",
        "lib/libmkl_sycl_lapack.so.5",
        "lib/libmkl_sycl_sparse.so.5",
        "lib/libmkl_sycl_dft.so.5",
        "lib/libmkl_sycl_vm.so.5",
        "lib/libmkl_sycl_rng.so.5",
        "lib/libmkl_sycl_stats.so.5",
        "lib/libmkl_sycl_data_fitting.so.5",
    ],
)

cc_library(
    name = "mkl_dpc",
    linkopts = [
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
