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

# SIMD sort library for x86-64: an x86-simd-sort release. This file is the source
# of truth for the release; dev/download_simd_sort.sh reads it too. Upstream
# packages install only the runtime dispatch header, so the root is a source tree
# and the headers come from `src/`.
#
# Renovate bumps _X86_VERSION; add the archive hash for the new version with:
#   curl -fsSL https://github.com/intel/x86-simd-sort/archive/refs/tags/<version>.tar.gz | sha256sum
_X86_VERSION = "v7.0"
_X86_SHA256 = {
    "v7.0": "f1658b0562a56255b157b7ffbb81e53e78e2571db1962262e55db6ca01733907",
}

simd_sort_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "SIMD_SORT_ROOT",
    # No archive without a hash, so an unhashed version fails when fetched rather
    # than being downloaded unverified.
    archives = [
        repos.archive(
            url = "https://github.com/intel/x86-simd-sort/archive/refs/tags/{}.tar.gz".format(_X86_VERSION),
            sha256 = _X86_SHA256[_X86_VERSION],
            strip_prefix = "x86-simd-sort-{}".format(_X86_VERSION.removeprefix("v")),
        ),
    ] if _X86_VERSION in _X86_SHA256 else [],
    includes = [
        "src",
    ],
    libs = [],
    build_template = "@onedal//dev/bazel/deps:simd_sort.tpl.BUILD",
)
