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

# OpenRNG (https://git.gitlab.arm.com/libraries/openrng) is a VSL-ABI-compatible
# RNG library used as an alternative to OpenBLAS's reference RNG on ARM (see
# dev/make/deps.ref.mk RNG_OPENRNG / .ci/env/openrng.sh). Like OpenBLAS, it is
# expected to be prebuilt and pointed to via OPENRNGROOT; Bazel does not build
# it from source.
openrng_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/libopenrng.a",
    ],
    build_template = "@onedal//dev/bazel/deps:openrng.tpl.BUILD",
)
