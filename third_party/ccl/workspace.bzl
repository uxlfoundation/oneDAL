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

"""External repo rule for oneCCL.

Repo-side symbol only. The ccl_test rule (a BUILD-time Bazel `rule()`) lives
in //dev/bazel/rules:ccl_test.bzl.
"""

load("@onedal//third_party:repo.bzl", "repos")

ccl_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include/cpu_gpu_dpcpp/oneapi/",
    ],
    libs = [
        "lib/cpu_gpu_dpcpp/libccl.a",
        "lib/cpu_gpu_dpcpp/libccl.so",
        "lib/cpu_gpu_dpcpp/libccl.so.1",
        "lib/cpu_gpu_dpcpp/libccl.so.1.0",
    ],
    build_template = "@onedal//third_party/ccl:ccl.tpl.BUILD",
)
