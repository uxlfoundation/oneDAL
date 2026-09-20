package(default_visibility = ["//visibility:public"])
load("@rules_cc//cc:defs.bzl", "cc_library")

# This template is used for non-Windows MKL packages. Windows uses mkl_win.tpl.BUILD.
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

# Classic (non-SYCL) MKL shared libraries. Kept in a target of their own so
# that the SYCL domain libraries below can declare a dependency on them: the
# oneDAL toolchain links with `-Wl,--as-needed`
# (dev/bazel/toolchains/cc_toolchain_lnx.bzl), where a shared library is only
# recorded as DT_NEEDED if it resolves a reference that is still undefined at
# the point the linker reads it. The SYCL domain libraries reference ~660
# classic MKL entry points (`cblas_*`, `mkl_lapack_*`, `mkl_serv_*`,
# `vsl*`, `mkl_sparse_*`), so the classic libraries must come *after* them on
# the link line. Expressing that as a Bazel dependency rather than as source
# order inside one target is what makes the ordering guaranteed: Bazel emits
# linker inputs in topological order, dependents first.
cc_library(
    name = "mkl_classic_binary",
    srcs = glob([
        "lib/libmkl_core.so*",
        "lib/libmkl_intel_lp64.so*",
        "lib/libmkl_gnu_thread.so*",
    ]),
)

cc_library(
    name = "mkl_dpc_utils",
    linkopts = [
        "-fsycl-max-parallel-link-jobs=16",
    ],
    srcs = glob([
        "lib/libmkl_sycl_blas.so*",
        "lib/libmkl_sycl_lapack.so*",
        "lib/libmkl_sycl_sparse.so*",
        "lib/libmkl_sycl_rng.so*",
    ]),
    deps = [
        ":mkl_classic_binary",
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

filegroup(
    name = "mkl_runtime",
    srcs = glob([
        "lib/libmkl*.so*",
    ], allow_empty = True),
)
