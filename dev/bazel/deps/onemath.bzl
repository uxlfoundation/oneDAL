#===============================================================================
# Copyright contributors to the oneDAL project
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

# oneMath (https://github.com/uxlfoundation/oneMath) implements the same DPC++
# interface as oneMKL and additionally dispatches to cuBLAS, cuSOLVER, cuSPARSE
# and cuRAND, which is what lets the oneDAL device sources run on NVIDIA GPUs.
#
# Unlike TBB or oneMKL there is no redistributable package to download: the
# backends a build needs are chosen at oneMath configure time
# (`-DENABLE_CUBLAS_BACKEND=ON` and friends), so the library has to be built
# locally and pointed at through ONEMATHROOT. Consequently this repository has
# no `urls` and only resolves when something actually depends on it, i.e. under
# `--dpc_math_backend=onemath`.
#
# oneDAL links only the run-time dispatching library, `libonemath.so`. The
# per-backend shared objects are `dlopen`ed by the dispatcher, so they are
# staged as optional libraries: they must sit next to `libonemath.so` at run
# time but are never link inputs, and which of them exist depends on how
# oneMath was configured.
#
# Every pattern is globbed the way `mkl.bzl` globs its shared objects, because
# oneMath installs `libonemath.so` as a symlink to the SONAME'd
# `libonemath.so.<abi>`. Staging only the symlink links fine -- `ld` follows it
# -- but records a `DT_NEEDED` on the SONAME that is then absent from the test
# runfiles, and every binary dies at startup with `libonemath.so.0: cannot open
# shared object file`. Bazel treats the versioned file as a runtime-only input,
# so naming both is exactly what is wanted.
onemath_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/libonemath.so*",
    ],
    optional_libs = [
        "lib/libonemath_blas_cublas.so*",
        "lib/libonemath_lapack_cusolver.so*",
        "lib/libonemath_rng_curand.so*",
        "lib/libonemath_sparse_blas_cusparse.so*",
        "lib/libonemath_blas_mklgpu.so*",
        "lib/libonemath_lapack_mklgpu.so*",
        "lib/libonemath_rng_mklgpu.so*",
    ],
    build_template = "@onedal//dev/bazel/deps:onemath.tpl.BUILD",
)
