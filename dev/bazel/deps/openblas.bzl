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
    includes = [
        "include",
    ],
    # oneDAL builds OpenBLAS with NO_FORTRAN=1 (see .ci/env/openblas.sh), so no
    # libgfortran.a is produced and none is needed — Make links only
    # libopenblas.a too (dev/make/deps.ref.mk). Listing it here created a
    # dangling symlink in the repo, which broke any action that consumes the
    # OpenBLAS archives, e.g. `cpp/daal/libonedal_thread.a`.
    libs = [
            "lib/libopenblas.a",
    ],
    build_template = "@onedal//dev/bazel/deps:openblas.tpl.BUILD",
)
