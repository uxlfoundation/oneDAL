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
        "lib/libmkl_sycl.so",
        "lib/libmkl_sycl_blas.so",
        "lib/libmkl_sycl_lapack.so",
        "lib/libmkl_sycl_sparse.so",
        "lib/libmkl_sycl_dft.so",
        "lib/libmkl_sycl_vm.so",
        "lib/libmkl_sycl_rng.so",
        "lib/libmkl_sycl_stats.so",
        "lib/libmkl_sycl_data_fitting.so",
        "lib/libmkl_sycl_blas.so.5",
        "lib/libmkl_sycl_lapack.so.5",
        "lib/libmkl_sycl_sparse.so.5",
        "lib/libmkl_sycl_dft.so.5",
        "lib/libmkl_sycl_vm.so.5",
        "lib/libmkl_sycl_rng.so.5",
        "lib/libmkl_sycl_stats.so.5",
        "lib/libmkl_sycl_data_fitting.so.5",
    ],
    build_template = "@onedal//dev/bazel/deps:mkl.tpl.BUILD",
    download_mapping = {
    # Required directory layout and layout in the downloaded
    # archives may be different. Mapping helps to setup relations
    # between required layout (LHS) and downloaded (RHS).
    # For example in this case, files from `lib/*` will be copied to `lib/intel64/*`.
    # "lib/intel64": "lib/",
    },
)
