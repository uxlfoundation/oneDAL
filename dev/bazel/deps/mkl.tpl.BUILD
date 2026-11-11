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
        # libmkl_intel_ilp64.a references mkl_serv_* symbols from libmkl_core.a.
        "lib/libmkl_intel_ilp64.a",
        "lib/libmkl_core.a",
        "lib/libmkl_tbb_thread.a",
    ],
    # The linkopts name the archives, and oneDAL's link rules pass only additional_inputs
    # on to a dependent link (`_unpack_linking_contexts` in cc/common.bzl).
    additional_linker_inputs = [
        "lib/libmkl_core.a",
        "lib/libmkl_intel_ilp64.a",
        "lib/libmkl_tbb_thread.a",
    ],
    linkopts = [
        # The archives are mutually recursive. `$(location)` keeps them execroot-relative,
        # so the link action does not depend on the output base and stays cacheable.
        "-Wl,--start-group",
        "$(location lib/libmkl_core.a)",
        "$(location lib/libmkl_intel_ilp64.a)",
        "$(location lib/libmkl_tbb_thread.a)",
        "-Wl,--end-group",
        # The TBB threading layer needs libstdc++. Named by file because clang
        # rewrites -lstdc++ to -lc++ under -stdlib=libc++.
        "-l:libstdc++.so.6",
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

# A target of its own so that, under --as-needed, it follows the SYCL domains that call
# into it on the link line (Bazel puts dependents first), as Make orders them.
cc_library(
    name = "mkl_classic_dynamic",
    srcs = glob([
        "lib/libmkl_core.so*",
        "lib/libmkl_intel_lp64.so*",
        "lib/libmkl_gnu_thread.so*",
    ]) + glob([
        # libmkl_core.so dlopens these kernels from its own directory, so they must share its
        # _solib dir; --as-needed keeps them off DT_NEEDED. The shipped ISA families vary.
        "lib/libmkl_def.so*",
        "lib/libmkl_avx*.so*",
        "lib/libmkl_mc*.so*",
        "lib/libmkl_vml_*.so*",
    ], allow_empty = True),
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
        ":mkl_classic_dynamic",
        "@openmp//:openmp_binary",
    ]
)

cc_library(
    name = "mkl_dpc",
    # TODO: take the parallel link job count from the command line.
    linkopts = [
        # Trades link speedup against resources; capped at `nproc` on smaller machines.
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
