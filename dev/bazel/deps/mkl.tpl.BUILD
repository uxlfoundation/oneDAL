package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    includes = [
        "include",
    ],
)

cc_library(
    name = "mkl_static",
    srcs = [
        # Keep MKL static archives in canonical dependency order.
        # libmkl_intel_ilp64.a references mkl_serv_* symbols from libmkl_core.a.
        "lib/libmkl_intel_ilp64.a",
        "lib/libmkl_core.a",
        "lib/libmkl_tbb_thread.a",
    ],
    linkopts = [
        # The source libraries have circular symbol dependencies. To successfully build this cc_library,
        # oneMKL requires wrapping the libraries with -Wl,--start-group and -Wl,--end-group.
        "-Wl,--start-group",
        "%{repo_root}/lib/libmkl_core.a",
        "%{repo_root}/lib/libmkl_intel_ilp64.a",
        "%{repo_root}/lib/libmkl_tbb_thread.a",
        "-Wl,--end-group",
        "-lpthread",
        "-lm",
        "-ldl",
    ],
    deps = [
        ":headers",
    ],
    defines = [
        "MKL_ILP64"
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
    srcs = glob([
        "lib/libmkl_core.so*",
        "lib/libmkl_intel_lp64.so*",
        "lib/libmkl_gnu_thread.so*",
        "lib/libmkl_sycl_blas.so*",
        "lib/libmkl_sycl_lapack.so*",
        "lib/libmkl_sycl_sparse.so*",
        "lib/libmkl_sycl_rng.so*",
    ]),
    deps = [
        "@openmp//:openmp_binary",
    ]
)

cc_library(
    name = "mkl_dpc",
    # TODO: add a mechanism to get attr from bazel command(it's not available for now)
    linkopts = [
        # Currently its hardcoded to 16 to get the best trade-off between linking speedup and resources used.
        # If the number of processors on machine is below 16 it will be defaulted to `nproc`.
        "-fsycl-max-parallel-link-jobs=16",
    ],
    deps = [
        ":headers",
        ":mkl_dpc_utils",
        "@opencl//:opencl_binary",
    ],
    defines = [
        "MKL_LP64"
    ],
)
