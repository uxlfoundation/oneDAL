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
load("@onedal//dev/bazel:utils.bzl", "paths")

def _get_dpcpp_compiler_root(repo_ctx):
    dpcpp_path = repo_ctx.which("icpx")
    return dpcpp_path and str(dpcpp_path.dirname.dirname.realpath)

def _opencl_repo_impl(repo_ctx):
    dpcpp_root = _get_dpcpp_compiler_root(repo_ctx)
    if dpcpp_root:
        lib_dir = paths.join(dpcpp_root, "lib")
        libs = [
            "libOpenCL.so",
            "libOpenCL.so.1",
            "libOpenCL.so.1.2",
        ]
        repos.create_symlinks(repo_ctx, lib_dir, libs)
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/deps:opencl.tpl.BUILD"),
    )

opencl_repo = repository_rule(
    implementation = _opencl_repo_impl,
    environ = [
        "PATH",
    ],
    local = True,
    configure = True,
)
