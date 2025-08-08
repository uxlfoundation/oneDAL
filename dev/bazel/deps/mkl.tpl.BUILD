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
