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

_VERSION = "2022.13.0"

dpl_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "DPL_ROOT",
    archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/68/aa/117ae55c433db29fa4f3f56a783181eddad1f3ac3dd3386e243a73d8af83/onedpl_devel-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "da07e1e9efbea6fe7a9dbdea0f0f5f346c2a7a3d1a3378be8c163235a9d19ee5",
            strip_prefix = "onedpl_devel-{}.data/data".format(_VERSION),
        ),
    ],
    includes = [
        "include",
    ],
    libs = [
        "lib",
    ],
    build_template = "@onedal//dev/bazel/deps:dpl.tpl.BUILD",
)
