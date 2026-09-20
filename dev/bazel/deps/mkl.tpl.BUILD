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
#
# Make links in the same order on Linux (`dev/make/deps.mkl.mk:91-95` lists the
# `sycl_blas`/`sycl_lapack`/`sycl_sparse`/`sycl_rng` domains before the classic
# libraries), so this is a parity fix, not a Bazel-specific workaround.
cc_library(
    name = "mkl_classic_binary",
    srcs = glob([
        "lib/libmkl_core.so*",
        "lib/libmkl_intel_lp64.so*",
        "lib/libmkl_gnu_thread.so*",
    ]) + glob([
        # CPU dispatch kernels. `libmkl_core.so.2` holds only the classic-MKL
        # dispatcher; the computational kernels live in per-ISA shared objects
        # that it `dlopen`s on the first classic-MKL call. It resolves them
        # against *its own* directory (`dladdr` on itself, plus its
        # `RPATH=$ORIGIN/.`), not against the consumer's RUNPATH, so they have
        # to be listed here, in the same `cc_library` as `libmkl_core.so.2`:
        # that is what makes Bazel symlink them into the same
        # `_solib_intel64/..._Ulib` directory and stage them in test runfiles.
        # A separate target would not work: it gets its own `_solib` directory,
        # which the dispatcher never looks in. `data` would not work either --
        # that reaches runfiles only, not the `_solib` directory the consumer
        # resolves `libmkl_core.so.2` from.
        #
        # They are deliberately not wanted as DT_NEEDED entries: mapping several
        # ISA variants at once is the kernel conflict MODULE.bazel warns about.
        # Nothing references their symbols statically, so the toolchain's
        # `-Wl,--as-needed` keeps them off DT_NEEDED while Bazel still stages
        # the files, leaving the dispatcher in sole control of what gets loaded.
        # Two consequences worth knowing: building with
        # `--linkopt=-Wl,--no-as-needed` does map every family at once, and if a
        # future MKL release exports a symbol a consumer references from a kernel
        # rather than from `libmkl_core.so.2`, that kernel becomes DT_NEEDED
        # silently. Both show up as DT_NEEDED differences against the Make build
        # in the release comparison (`dev/release_tests`).
        #
        # Without them every target that reaches classic MKL at run time dies
        # with
        #   INTEL oneMKL ERROR: .../libmkl_avx512.so.2: cannot open shared
        #     object file: No such file or directory.
        #   Intel oneMKL FATAL ERROR: Cannot load libmkl_avx512.so.2 or
        #     libmkl_def.so.2.
        #
        # `allow_empty` because `repos.bzl` symlinks these from `optional_libs`:
        # which ISA families an MKL package ships changes between releases, and a
        # package missing one must still build.
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
