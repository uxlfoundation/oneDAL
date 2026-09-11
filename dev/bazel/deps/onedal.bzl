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

onedal_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        # Static
        "lib/intel64/libonedal_core.a",
        "lib/intel64/libonedal_thread.a",
        "lib/intel64/libonedal.a",
        "lib/intel64/libonedal_parameters.a",

        # Dynamic
        "lib/intel64/libonedal_core.so",
        "lib/intel64/libonedal_core.so.%{version_binary_major}",
        "lib/intel64/libonedal_core.so.%{version_binary_major}.%{version_binary_minor}",
        "lib/intel64/libonedal_thread.so",
        "lib/intel64/libonedal_thread.so.%{version_binary_major}",
        "lib/intel64/libonedal_thread.so.%{version_binary_major}.%{version_binary_minor}",
        "lib/intel64/libonedal.so",
        "lib/intel64/libonedal.so.%{version_binary_major}",
        "lib/intel64/libonedal.so.%{version_binary_major}.%{version_binary_minor}",
        "lib/intel64/libonedal_dpc.so",
        "lib/intel64/libonedal_dpc.so.%{version_binary_major}",
        "lib/intel64/libonedal_dpc.so.%{version_binary_major}.%{version_binary_minor}",
        "lib/intel64/libonedal_parameters.so",
        "lib/intel64/libonedal_parameters.so.%{version_binary_major}",
        "lib/intel64/libonedal_parameters.so.%{version_binary_major}.%{version_binary_minor}",
        "lib/intel64/libonedal_parameters_dpc.so",
        "lib/intel64/libonedal_parameters_dpc.so.%{version_binary_major}",
        "lib/intel64/libonedal_parameters_dpc.so.%{version_binary_major}.%{version_binary_minor}",
    ],
    # Globbed rather than listed one by one so that a release tree built with
    # the debug MSVC runtime — where every library carries a `d` suffix
    # (`onedal_cored.lib`), see dev/bazel/cc.bzl `_msvc_runtime_suffix` — is
    # picked up as well, including a combined //:release_all tree holding both.
    # `onedal_win.tpl.BUILD` names the individual files in its `select()`, so
    # a flavour missing from the tree still fails with a clear missing-input
    # error rather than being silently skipped.
    win_libs = [
        # Static libraries and dynamic import libraries.
        "lib/intel64/onedal*.lib",
    ],
    win_bins = [
        "redist/intel64/onedal*.dll",
    ],
    build_template = "@onedal//dev/bazel/deps:onedal.tpl.BUILD",
    win_build_template = "@onedal//dev/bazel/deps:onedal_win.tpl.BUILD",
)
