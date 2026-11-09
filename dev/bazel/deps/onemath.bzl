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
# per-backend shared objects are `dlopen`ed by the dispatcher through a
# `$ORIGIN` RUNPATH, which under Bazel is the `_solib` directory and not the
# oneMath install, so they are found through `LD_LIBRARY_PATH` (see
# `--config=nvidia-gpu` in dev/bazel/README.md) and are not staged here.
#
# Only the SONAME'd `libonemath.so.<abi>` is staged and linked, as with
# `libtbb.so.12`: it is what `DT_NEEDED` names, so it must be in the runfiles,
# and staging the `libonemath.so` symlink as well would link the library twice.
onemath_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/libonemath.so.*",
    ],
    build_template = "@onedal//dev/bazel/deps:onemath.tpl.BUILD",
)
