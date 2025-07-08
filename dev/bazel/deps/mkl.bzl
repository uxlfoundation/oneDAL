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

mkl_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
        "lib/libmkl_core.a",
        "lib/libmkl_intel_ilp64.a",
        "lib/libmkl_tbb_thread.a",
        "lib/libmkl_sycl.a",
    ],
    build_template = "@onedal//dev/bazel/deps:mkl.tpl.BUILD",
    download_mapping = {
    # Required directory layout and layout in the downloaded
    # archives may be different. Mapping helps to setup relations
    # between required layout (LHS) and downloaded (RHS).
    # In this case, files from `lib/*` will be copied to `lib/intel64/*`.
    "lib/intel64": "lib/",
    },
)
