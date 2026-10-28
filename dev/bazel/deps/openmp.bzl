#===============================================================================
# Copyright 2026 Intel Corporation
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

# GNU OpenMP (libgomp), required by the MKL `gnu_thread` layer.
_VERSION = "14.2.0"

openmp_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "GOMPROOT",
    archives = [
        repos.archive(
            url = "https://anaconda.org/conda-forge/libgomp/{v}/download/linux-64/libgomp-{v}-h77fa898_1.conda".format(v = _VERSION),
            sha256 = "1911c29975ec99b6b906904040c855772ccb265a1c79d5d75c8ceec4ed89cd63",
            # The .conda package already unpacks into the required layout.
            strip_prefix = "",
        ),
    ],
    includes = [
        "include",
    ],
    libs = [
        "lib/libgomp.so*",
    ],
    build_template = "@onedal//dev/bazel/deps:openmp.tpl.BUILD",
)
