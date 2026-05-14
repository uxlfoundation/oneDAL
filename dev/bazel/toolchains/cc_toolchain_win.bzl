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

load("@rules_cc//cc/private/toolchain:lib_cc_configure.bzl",
    "get_starlark_list",
)
load("@onedal//dev/bazel:utils.bzl", "paths")
load("@onedal//dev/bazel/toolchains:common.bzl",
    "TEST_CPP_FILE",
    "add_compiler_option_if_supported",
    "get_cpu_specific_options",
    "get_default_compiler_options",
    "get_starlark_list_dict",
    "get_toolchain_identifier",
)

def _find_tool(repo_ctx, tool_name, mandatory = False):
    tool_path = repo_ctx.which(tool_name)
    is_found = tool_path != None
    if not is_found:
        if mandatory:
            fail("Cannot find {}; is it on PATH?".format(tool_name))
        else:
            # Stub path so repo generation does not fail; the resulting
            # toolchain is only valid when the compiler is actually present.
            repo_ctx.template(
                "tool_not_found_{}.bat".format(tool_name),
                Label("@onedal//dev/bazel/toolchains/tools:tool_not_found.tpl.bat"),
                {"%{tool_name}": tool_name},
            )
            tool_path = repo_ctx.path("tool_not_found_{}.bat".format(tool_name))
    return str(tool_path), is_found

def _find_tools_icx(repo_ctx):
    cc_path, _ = _find_tool(repo_ctx, "icx", mandatory = True)
    dpcc_path, dpcpp_found = _find_tool(repo_ctx, "icpx", mandatory = False)
    # On Windows the Intel compiler suite ships `llvm-lib` and `lld-link`
    # next to icx.exe. Both accept MSVC-style /OUT: and /LIBPATH: syntax.
    ar_path, _ = _find_tool(repo_ctx, "llvm-lib", mandatory = False)
    if not ar_path or ar_path.endswith("tool_not_found_llvm-lib.bat"):
        # Fall back to lib.exe shipped by MSVC (required by oneAPI install).
        ar_path, _ = _find_tool(repo_ctx, "lib", mandatory = True)
    return struct(
        cc = cc_path,
        dpcc = dpcc_path,
        # icx/icpx drive the link step; no separate wrapper script needed on
        # Windows — lld-link is invoked implicitly via `-fuse-ld=lld-link`.
        cc_link = cc_path,
        dpcc_link = dpcc_path,
        ar = ar_path,
        is_dpc_found = dpcpp_found,
    )

def _is_icx_requested(repo_ctx):
    """Return True if the Windows toolchain should use Intel oneAPI icx/icpx.

    Default behavior: icx is selected whenever it is on PATH. Users can
    opt out via ONEDAL_WIN_COMPILER=cl (or CC=cl) to fall back to the
    rules_cc MSVC auto-config.
    """
    override = repo_ctx.os.environ.get("ONEDAL_WIN_COMPILER", "").lower()
    if override == "cl":
        return False
    if override == "icx":
        return True
    cc = repo_ctx.os.environ.get("CC", "").lower()
    if cc == "cl" or cc.endswith("\\cl.exe") or cc.endswith("/cl.exe"):
        return False
    # Fall back to icx if the compiler is available.
    return repo_ctx.which("icx") != None

def _configure_cc_toolchain_win_cl(repo_ctx):
    """Default Windows toolchain: delegate to rules_cc's MSVC cl auto-config."""
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:cc_toolchain_win.tpl.BUILD"),
    )

def _configure_cc_toolchain_win_icx(repo_ctx, reqs):
    """Custom Windows toolchain using Intel oneAPI icx/icpx."""
    repo_ctx.file(TEST_CPP_FILE, "int main() { return 0; }")

    tools = _find_tools_icx(repo_ctx)

    # icx/icpx accept clang-style include detection via `-E -v`.
    # We leave the builtin include discovery empty for now — Bazel will
    # treat system headers reached via INCLUDE env as implicit. Populating
    # this list reliably requires parsing the `-v` output which differs from
    # the Linux cc path; punt until we have a real environment to test in.
    builtin_include_directories = []

    reqs_icx = struct(
        os_id = "win",
        compiler_id = "icx",
        dpc_compiler_id = "icpx",
        compiler_version = "local",
        dpc_compiler_version = "local",
        target_arch_id = "intel64",
        host_arch_id = "intel64",
        libc_version = "local",
        libc_abi_version = "local",
        compiler_abi_version = "local",
    )

    dpcc_code_split = "per_kernel"

    compile_flags_cc = get_default_compiler_options(
        repo_ctx, reqs_icx, tools.cc,
        is_dpcc = False, category = "common",
    )
    compile_flags_dpcc = get_default_compiler_options(
        repo_ctx, reqs_icx, tools.dpcc,
        is_dpcc = True, category = "common",
    ) if tools.is_dpc_found else []
    if tools.is_dpc_found:
        compile_flags_dpcc = compile_flags_dpcc + add_compiler_option_if_supported(
            repo_ctx, tools.dpcc,
            "-fsycl-device-code-split={}".format(dpcc_code_split),
        )

    compile_flags_pedantic_cc = get_default_compiler_options(
        repo_ctx, reqs_icx, tools.cc,
        is_dpcc = False, category = "pedantic",
    )
    compile_flags_pedantic_dpcc = get_default_compiler_options(
        repo_ctx, reqs_icx, tools.dpcc,
        is_dpcc = True, category = "pedantic",
    ) if tools.is_dpc_found else []

    # Force clang/GCC driver mode on Windows: icx.exe can default to clang-cl
    # parsing, while this toolchain emits clang/GCC-style compile and link flags.
    # Route linking through lld-link so clang-style -Wl,... passthrough works.
    common_link_flags = ["--driver-mode=g++", "-fuse-ld=lld-link"]

    link_flags_cc = common_link_flags
    link_flags_dpcc = (common_link_flags + [
        "-fsycl",
        "-fsycl-device-code-split={}".format(dpcc_code_split),
    ]) if tools.is_dpc_found else []

    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:cc_toolchain_win_icx.tpl.BUILD"),
        {
            "%{cc_toolchain_identifier}": get_toolchain_identifier(reqs_icx),
            "%{compiler}": "icx-local",
            "%{abi_version}": "local",
            "%{abi_libc_version}": "local",
            "%{target_libc}": "local",
            "%{target_cpu}": "intel64",
            "%{host_system_name}": "win-intel64",
            "%{target_system_name}": "win-intel64",
            "%{supports_param_files}": "1",
            "%{compiler_deps}": get_starlark_list([]),
            "%{ar_deps}": get_starlark_list([]),
            "%{linker_deps}": get_starlark_list([]),
            "%{cc_path}": tools.cc,
            "%{dpcc_path}": tools.dpcc,
            "%{cc_link_path}": tools.cc_link,
            "%{dpcc_link_path}": tools.dpcc_link,
            "%{ar_path}": tools.ar,
            "%{cxx_builtin_include_directories}": get_starlark_list(
                builtin_include_directories,
            ),
            "%{compile_flags_cc}": get_starlark_list(compile_flags_cc),
            "%{compile_flags_dpcc}": get_starlark_list(compile_flags_dpcc),
            "%{compile_flags_pedantic_cc}": get_starlark_list(compile_flags_pedantic_cc),
            "%{compile_flags_pedantic_dpcc}": get_starlark_list(compile_flags_pedantic_dpcc),
            "%{cxx_flags}": get_starlark_list([]),
            "%{link_flags_cc}": get_starlark_list(link_flags_cc),
            "%{link_flags_dpcc}": get_starlark_list(link_flags_dpcc),
            "%{dynamic_link_libs}": get_starlark_list([]),
            "%{opt_compile_flags}": get_starlark_list([
                "-O2",
                "-DNDEBUG",
            ]),
            "%{opt_link_flags}": get_starlark_list([]),
            "%{dbg_compile_flags}": get_starlark_list([
                "-g",
                "-O1",
                "-DONEDAL_ENABLE_ASSERT",
                "-DDEBUG_ASSERT",
            ]),
            "%{deterministic_compile_flags}": get_starlark_list([
                "-Wno-builtin-macro-redefined",
                "-D__DATE__=\\\"redacted\\\"",
                "-D__TIMESTAMP__=\\\"redacted\\\"",
                "-D__TIME__=\\\"redacted\\\"",
            ]),
            "%{cpu_flags_cc}": get_starlark_list_dict(
                get_cpu_specific_options(reqs_icx),
            ),
            "%{cpu_flags_dpcc}": get_starlark_list_dict(
                get_cpu_specific_options(reqs_icx, is_dpcc = True),
            ),
        },
    )

def configure_cc_toolchain_win(repo_ctx, reqs):
    if _is_icx_requested(repo_ctx):
        _configure_cc_toolchain_win_icx(repo_ctx, reqs)
    else:
        _configure_cc_toolchain_win_cl(repo_ctx)
