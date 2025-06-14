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
libs = select({
    "@platforms//os:windows": [
        # Static libraries
        "lib/intel64/onedal_core.lib",
        "lib/intel64/onedal_thread.lib",
        "lib/intel64/onedal.lib",
        "lib/intel64/onedal_dpc.lib",
        "lib/intel64/onedal_parameters.lib",
        "lib/intel64/onedal_parameters_dpc.lib",

        # Dynamic import libraries (если есть .dll, нужна соответствующая .lib)
        "lib/intel64/onedal_core_dll.lib",
        "lib/intel64/onedal_thread_dll.lib",
        "lib/intel64/onedal_dll.lib",
        "lib/intel64/onedal_dpc_dll.lib",
        "lib/intel64/onedal_parameters_dll.lib",
        "lib/intel64/onedal_parameters_dpc_dll.lib",
    ],
    "//conditions:default": [
        # Static libraries
        "lib/intel64/libonedal_core.a",
        "lib/intel64/libonedal_thread.a",
        "lib/intel64/libonedal.a",
        "lib/intel64/libonedal_dpc.a",
        "lib/intel64/libonedal_parameters.a",
        "lib/intel64/libonedal_parameters_dpc.a",

        # Dynamic libraries (optional, only if needed for runtime)
        "lib/intel64/libonedal_core.so",
        "lib/intel64/libonedal_thread.so",
        "lib/intel64/libonedal.so",
        "lib/intel64/libonedal_dpc.so",
        "lib/intel64/libonedal_parameters.so",
        "lib/intel64/libonedal_parameters_dpc.so",
    ],
}),
    build_template = "@onedal//dev/bazel/deps:onedal.tpl.BUILD",
)
