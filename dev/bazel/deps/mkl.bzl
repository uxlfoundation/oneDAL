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

# Pinned in requirements_lock.txt (Renovate); keep all oneMKL packages on one version.
mkl_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "MKLROOT",
    pypi_packages = [
        # Static for the host build; the DPC build uses the dynamic classic MKL.
        "mkl-static",
        "mkl",
        "mkl-include",
        "onemkl-sycl-include",
        # These wheels ship only the versioned `.so.N`, which mkl.tpl.BUILD globs.
        "onemkl-sycl-blas",
        "onemkl-sycl-lapack",
        "onemkl-sycl-sparse",
        "onemkl-sycl-rng",
    ],
    # Windows CPU/MSVC builds; Windows DPC++/SYCL dependencies are out of scope.
    win_pypi_packages = [
        "mkl-static",
        "mkl",
        "mkl-devel",
        "mkl-include",
    ],
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
    # VML kernels are dlopened like the dispatch kernels, but no oneDAL target is known to need one.
    optional_libs = [
        "lib/libmkl_vml_*.so*",
    ],
    # The ISA families vary between releases, but a package with no dispatch kernel fails at the
    # first MKL call, so fail here instead. On Windows the required `bin/*.dll` covers them.
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
        # Also the debug-CRT `mkl_tbb_threadd.lib`; mkl_win.tpl.BUILD names each variant.
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
)
