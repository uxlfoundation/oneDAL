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

load("@onedal//dev/bazel/deps:gcc.bzl", "download_gcc")
load("@onedal//dev/bazel/toolchains:common.bzl", "detect_os", "detect_compiler", "detect_host_arch", "detect_target_arch")
load("@onedal//dev/bazel/toolchains:cc_toolchain_lnx.bzl", "configure_cc_toolchain_lnx")
load("@onedal//dev/bazel/toolchains:cc_toolchain_win.bzl", "configure_cc_toolchain_win")

def _needs_downloaded_compiler(repo_ctx, os_id, host_arch_id):
    if os_id != "lnx" or host_arch_id != "intel64" or "CC" in repo_ctx.os.environ:
        return False
    if repo_ctx.os.environ.get("ONEDAL_DOWNLOAD_CC") == "1":
        return True
    return not any([repo_ctx.which(cc) for cc in ["icx", "gcc", "clang"]])

def _detect_requirements(repo_ctx):
    os_id = detect_os(repo_ctx)
    compiler_id = detect_compiler(repo_ctx, os_id)
    host_arch_id = detect_host_arch(repo_ctx, os_id)
    target_arch_id = detect_target_arch(repo_ctx, host_arch_id)
    downloaded_tools = None
    if _needs_downloaded_compiler(repo_ctx, os_id, host_arch_id):
        compiler_id = "gcc"
        downloaded_tools = download_gcc(repo_ctx, "gcc")
    return struct(
        os_id = os_id,
        compiler_id = compiler_id,

        # Paths of the downloaded compiler and binutils, or None to look them up.
        downloaded_tools = downloaded_tools,

        libc_version = "local",
        libc_abi_version = "local",
        compiler_abi_version = "local",

        host_arch_id = host_arch_id,
        target_arch_id = target_arch_id,

        compiler_version = "local",
        dpc_compiler_id = "icpx",
    )

def _configure_cc_toolchain(repo_ctx, reqs):
    configure_cc_toolchain_os = {
        "lnx": configure_cc_toolchain_lnx,
        "win": configure_cc_toolchain_win,
    }[reqs.os_id]
    return configure_cc_toolchain_os(repo_ctx, reqs)

def _onedal_cc_toolchain_impl(repo_ctx):
    reqs = _detect_requirements(repo_ctx)
    _configure_cc_toolchain(repo_ctx, reqs)

onedal_cc_toolchain = repository_rule(
    implementation = _onedal_cc_toolchain_impl,
    environ = [
        "CC",
        "CXX",
        "PATH",
        # =1 downloads the GNU toolchain even when a compiler is in PATH.
        "ONEDAL_DOWNLOAD_CC",
        "INCLUDE",
        "LIB",
        # icx or clang; anything else selects the rules_cc MSVC config. Windows ARM64 is clang-cl only.
        "ONEDAL_WIN_COMPILER",
        # The Visual Studio rules_cc configures for cl: the one vcvarsall.bat
        # set up (VCINSTALLDIR), unless BAZEL_VC or BAZEL_VS names another.
        "BAZEL_VC",
        "BAZEL_VS",
        "VCINSTALLDIR",
        # Windows host-arch detection (see detect_host_arch in common.bzl); the
        # toolchain repo must re-evaluate when these change.
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_ARCHITEW6432",
    ],
)
