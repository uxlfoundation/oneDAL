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

"""The Windows resource compiler, provided by the toolchain instead of looked up in PATH.

The Windows toolchain repository configures it at repository time (`llvm-rc` for clang-cl,
`rc.exe` for icx and MSVC) and registers it under `RC_TOOLCHAIN_TYPE`. Linux registers none,
so the type is optional for rules that compile resources.
"""

RC_TOOLCHAIN_TYPE = "@onedal//dev/bazel/toolchains:rc_toolchain_type"

def _rc_toolchain_impl(ctx):
    return [platform_common.ToolchainInfo(
        rc_path = ctx.attr.rc_path,
        files = ctx.files.files,
    )]

rc_toolchain = rule(
    implementation = _rc_toolchain_impl,
    attrs = {
        "rc_path": attr.string(
            doc = "Absolute path of the resource compiler; empty when none was found at configure time.",
        ),
        "files": attr.label_list(
            allow_files = True,
            doc = "Files the compiler's actions take as inputs, such as the toolchain's tool identity.",
        ),
    },
)
