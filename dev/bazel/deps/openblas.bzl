#===============================================================================
# Copyright 2023 Intel Corporation
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

openblas_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "OPENBLASROOT",
    includes = [
        "include",
    ],
    # No libgfortran.a: .ci/env/openblas.sh builds with NO_FORTRAN=1, and Make links only this.
    libs = [
            "lib/libopenblas.a",
    ],
    build_template = "@onedal//dev/bazel/deps:openblas.tpl.BUILD",
    # `.ci/env/openblas.bat` builds a static OpenBLAS, so no `win_bins` (Make: OPENBLASDIR.libia.win).
    win_libs = [
        "lib/openblas.lib",
    ],
    win_build_template = "@onedal//dev/bazel/deps:openblas_win.tpl.BUILD",
)
