#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

load("@onedal//dev/bazel:repos.bzl", "repos")

mkl_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/libmkl_core.a",
        "lib/libmkl_intel_ilp64.a",
        "lib/libmkl_tbb_thread.a",
        "lib/libmkl_core.so*",
        "lib/libmkl_intel_lp64.so*",
        "lib/libmkl_gnu_thread.so*",
        "lib/libmkl_sycl_blas.so*",
        "lib/libmkl_sycl_lapack.so*",
        "lib/libmkl_sycl_sparse.so*",
        "lib/libmkl_sycl_rng.so*",

    ],
    # VML kernels are `dlopen`-ed like the CPU dispatch kernels below and the
    # pinned package ships them (`libmkl_vml_def`, `_avx2`, `_avx512`, `_mc3`,
    # `_cmpt`), so they are symlinked when present. They stay merely optional
    # rather than collectively required because the failure measured below is the
    # dispatch kernels': no oneDAL target was observed to need a VML kernel, and
    # requiring a file no measurement implicates would be a guess.
    optional_libs = [
        "lib/libmkl_vml_*.so*",
    ],
    # CPU dispatch kernels. `libmkl_core.so.2` holds only the dispatcher: the
    # actual kernels live in per-ISA shared objects that it `dlopen`s by SONAME
    # at the first classic-MKL call. Without any of them in the repository every
    # target that reaches classic MKL at run time dies with
    #   INTEL oneMKL ERROR: .../libmkl_avx512.so.2: cannot open shared object
    #     file
    #   Intel oneMKL FATAL ERROR: Cannot load libmkl_avx512.so.2 or
    #     libmkl_def.so.2.
    # Which ISA families a given release ships changes between versions (the
    # pinned 2025 package has no `libmkl_avx.so*`/`libmkl_mc.so*`), so no single
    # family can be required -- that would break host-only builds on a perfectly
    # good package. But *none* of them is not a layout to build against, so the
    # set is collectively required: present families are symlinked, and a package
    # with the classic libraries and no kernel fails here, naming the package,
    # rather than at the first `gemm` call, naming a file the user never asked
    # for. They end up in the same `cc_library` as `libmkl_core.so.2`
    # (`mkl_classic_dynamic` in mkl.tpl.BUILD), which is where the dispatcher
    # looks; see the comment there.
    #
    # Windows needs no counterpart: `win_bins` globs `bin/*.dll`, which is a
    # required pattern and covers `mkl_def.2.dll` and the ISA kernels with it.
    required_any_libs = [
        "lib/libmkl_def.so*",
        "lib/libmkl_avx*.so*",
        "lib/libmkl_mc*.so*",
    ],
    required_any_libs_description = "a classic MKL CPU dispatch kernel",
    build_template = "@onedal//dev/bazel/deps:mkl.tpl.BUILD",
    win_includes = [
        "include",
    ],
    win_libs = [
        "lib/mkl_core.lib",
        "lib/mkl_intel_ilp64.lib",
        # Globbed so the debug-CRT threading layer (`mkl_tbb_threadd.lib`,
        # see dev/make/deps.mkl.mk:51) is symlinked too when the MKL layout
        # ships it. `mkl_win.tpl.BUILD` names both variants explicitly in its
        # select(), so a debug-runtime build with the file absent fails with a
        # plain missing-input error instead of silently linking the release one.
        "lib/mkl_tbb_thread*.lib",
        "lib/mkl_core_dll.lib",
        "lib/mkl_intel_lp64_dll.lib",
        "lib/mkl_intel_thread_dll.lib",
        "lib/mkl_sycl_blas_dll.lib",
        "lib/mkl_sycl_lapack_dll.lib",
        "lib/mkl_sycl_rng_dll.lib",
        "lib/mkl_sycl_sparse_dll.lib",
    ],
    win_bins = [
        "bin/*.dll",
    ],
    win_build_template = "@onedal//dev/bazel/deps:mkl_win.tpl.BUILD",
    download_mapping = {
    # Required directory layout and layout in the downloaded
    # archives may be different. Mapping helps to setup relations
    # between required layout (LHS) and downloaded (RHS).
    # For example in this case, files from `lib/*` will be copied to `lib/intel64/*`.
    # "lib/intel64": "lib/",
    },
)
