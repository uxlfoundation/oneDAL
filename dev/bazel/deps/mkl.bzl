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

# Building and linking needs the unversioned `.so` symlinks next to the
# versioned files (`libmkl_blas_sycl.so.5`). The PyPI packages ship only the
# latter, so the SYCL domain libraries are taken from Anaconda, which ships
# both. Headers, for which the distinction does not matter, stay on PyPI.
_VERSION = "2025.2.0"

# The dynamic core package is pinned separately from the rest, which are kept in
# lockstep on `_VERSION`.
_CORE_VERSION = "2025.3.0"

# Windows CPU/MSVC builds use the latest classic MKL PyPI wheels; Windows
# DPC++/SYCL dependencies are out of scope.
_WIN_VERSION = "2026.0.0"

mkl_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "MKLROOT",
    archives = [
        # Core static MKL runtime libraries (libmkl_intel_lp64.a, libmkl_core.a,
        # libmkl_sequential.a, ...). The dynamic MKL core cannot be used from
        # Bazel: the dispatcher's avx512/avx2/cm3/def symbols conflict inside a
        # cc_library.
        # TODO: investigate how to provide global runtime dependencies in Bazel.
        repos.archive(
            url = "https://anaconda.org/conda-forge/mkl-static/{v}/download/linux-64/mkl-static-{v}-ha770c72_629.conda".format(v = _VERSION),
            sha256 = "a982243984e68abc692ad1c855902163f646d0c2121e8821c54c28e77f1939ec",
            # The .conda packages already unpack into the required layout.
            strip_prefix = "",
        ),
        # Headers of the classic C/C++ interface (`mkl.h` and friends). Does not
        # follow the oneAPI directory structure.
        repos.archive(
            url = "https://files.pythonhosted.org/packages/11/58/6f583b3bac7d3952a89a00ab34e61baa17f6d6de3454a8005958289bef22/mkl_include-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "691ceaccf6d960e19d47304d24ca2ee4e807810077e93c1c86c2e32cd6223012",
            strip_prefix = "mkl_include-{}.data/data".format(_VERSION),
        ),
        # Headers of the DPC++/SYCL interface (`onemkl/*.hpp`).
        repos.archive(
            url = "https://files.pythonhosted.org/packages/67/60/183badc2d807be1abb95a20315e84a2075cb44a1d1ede104d42cb1ed3092/onemkl_sycl_include-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "4e995c02e5f43265aa830a06e538b2e5ada76b7c2785c26b788d6073ba605b0f",
            strip_prefix = "onemkl_sycl_include-{}.data/data".format(_VERSION),
        ),
        # SYCL BLAS domain: libmkl_sycl_blas.so.5 (GEMM, AXPY, ...).
        repos.archive(
            url = "https://anaconda.org/conda-forge/onemkl-sycl-blas/{v}/download/linux-64/onemkl-sycl-blas-{v}-haf5e11a_628.conda".format(v = _VERSION),
            sha256 = "0748e4f91e328f4ae995013b10ae79966f80cf62a72e76495dfe1841cd022235",
            strip_prefix = "",
        ),
        # SYCL LAPACK domain: libmkl_sycl_lapack.so.5 (LU, QR, eigenvalues).
        repos.archive(
            url = "https://anaconda.org/conda-forge/onemkl-sycl-lapack/{v}/download/linux-64/onemkl-sycl-lapack-{v}-haf5e11a_628.conda".format(v = _VERSION),
            sha256 = "e56ec1e4e0fbfa28783a28e99337ca541934cb3aaaa0ecca469a27dab2f3a762",
            strip_prefix = "",
        ),
        # SYCL sparse domain: libmkl_sycl_sparse.so.5.
        repos.archive(
            url = "https://anaconda.org/conda-forge/onemkl-sycl-sparse/{v}/download/linux-64/onemkl-sycl-sparse-{v}-haf5e11a_628.conda".format(v = _VERSION),
            sha256 = "b2cdba1894464927a05115c04bcc1a5b64e25c116c45ae44c75990c101162b35",
            strip_prefix = "",
        ),
        # SYCL RNG domain: libmkl_sycl_rng.so.5.
        repos.archive(
            url = "https://anaconda.org/conda-forge/onemkl-sycl-rng/{v}/download/linux-64/onemkl-sycl-rng-{v}-haf5e11a_628.conda".format(v = _VERSION),
            sha256 = "e5664adaca1c2e15a771b0d67ec7691137c8ba8e0a5a6af6fdffece9ae37692e",
            strip_prefix = "",
        ),
        # Dynamic core: libmkl_core.so.2, required for SYCL execution.
        repos.archive(
            url = "https://anaconda.org/conda-forge/mkl/{v}/download/linux-64/mkl-{v}-h0e700b2_463.conda".format(v = _CORE_VERSION),
            sha256 = "659d79976f06d2b796a0836414573a737a0856b05facfa77e5cc114081a8b3d4",
            strip_prefix = "",
        ),
    ],
    win_archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/97/b8/232e453647e55d00f51d57a6b0f5d7da575188ff392d850b490ee1993b26/mkl_static-{}-py2.py3-none-win_amd64.whl".format(_WIN_VERSION),
            sha256 = "c0d5df2e923836a07e2f2f0db95ead67ae91f98559f5215fb82add910bb15732",
            strip_prefix = "mkl_static-{}.data/data/Library".format(_WIN_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/7b/2f/8a77106ef648cc1f04aa27e8b6c50cd82de38d1767801ef29cd75b8026e8/mkl-{}-py2.py3-none-win_amd64.whl".format(_WIN_VERSION),
            sha256 = "801c9cc748be81d8269bfec2495d02be38560fc7386439c18499ab3f32b06b3e",
            strip_prefix = "mkl-{}.data/data/Library".format(_WIN_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/3b/32/7c214676e0707dcdb0709961a63bdd010af34ff9ead46d3bae38e36102c6/mkl_devel-{}-py2.py3-none-win_amd64.whl".format(_WIN_VERSION),
            sha256 = "0d74e49b79ca10a332f4e4768afc597657fe9ce3d76f18bf7737d1a5043148a3",
            strip_prefix = "mkl_devel-{}.data/data/Library".format(_WIN_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/2d/4f/1944dfa73b63c183c5a48da47d7b140d97146d0a96a0e7ef8c2bd8f0739e/mkl_include-{}-py2.py3-none-win_amd64.whl".format(_WIN_VERSION),
            sha256 = "8e316479bc3a5cccb1161e8de2140e476bd5a8ecad840374770d529b591d8af6",
            strip_prefix = "mkl_include-{}.data/data/Library".format(_WIN_VERSION),
        ),
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
)
