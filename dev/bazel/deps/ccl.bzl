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
load("@onedal//dev/bazel/deps:mpi.bzl", "mpi_test")

ccl_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "CCL_ROOT",
    # Pinned in requirements_lock.txt; the runtime wheel ships no headers.
    pypi_packages = [
        "oneccl",
        "oneccl-devel",
    ],
    includes = [
        "include/oneapi",
    ],
    libs = [
        "lib/libccl.so",
        "lib/libccl.so.1",
        "lib/libccl.so.1.0",
    ],
    build_template = "@onedal//dev/bazel/deps:ccl.tpl.BUILD",
    # oneCCL installations before 2021.14 keep both under `cpu_gpu_dpcpp/`.
    fallback_mapping = {
        "lib/":                                         "lib/cpu_gpu_dpcpp/",
        "include/":                                     "include/cpu_gpu_dpcpp/",
    },
)

# A CCL test is launched the same way, through Intel MPI's mpiexec.
ccl_test = mpi_test
