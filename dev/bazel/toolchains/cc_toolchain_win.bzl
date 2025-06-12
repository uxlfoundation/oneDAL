# Copyright 2025, adapted from Intel Corporation's Linux toolchain
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

load(
    "@rules_cc//cc/private/toolchain:lib_cc_configure.bzl",
    "auto_configure_fail",
    "get_starlark_list",
    "write_builtin_include_directory_paths",
)
load(
    "@onedal//dev/bazel:utils.bzl",
    "paths",
    "utils",
)
load(
    "@onedal//dev/bazel/toolchains:common.bzl",
    "TEST_CPP_FILE",
    "add_compiler_option_if_supported",
    "add_linker_option_if_supported",
    "get_cpu_specific_options",
    "get_cxx_inc_directories",
    "get_default_compiler_options",
    "get_no_canonical_prefixes_opt",
    "get_starlark_list_dict",
    "get_toolchain_identifier",
    "get_tmp_dpcpp_inc_directories",
)

def _find_tool(repo_ctx, tool_name, mandatory = False):
    if tool_name.startswith("\\") or tool_name[1:3] == ":\\":
        return tool_name
    tool_path = repo_ctx.which(tool_name)
    is_found = tool_path != None
    if not is_found:
        if mandatory:
            auto_configure_fail("Cannot find {}; ensure it is in your %PATH%".format(tool_name))
        else:
            repo_ctx.template(
                "tool_not_found.bat",
                Label("@onedal//dev/bazel/toolchains/tools:tool_not_found.tpl.bat"),
                {"%{tool_name}": tool_name},
            )
            tool_path = repo_ctx.path("tool_not_found.bat")
    return str(tool_path), is_found

def find_tool(repo_ctx, tool_name, mandatory = False):
    return _find_tool(repo_ctx, tool_name, mandatory)

def _create_lib_merge_tool(repo_ctx, lib_path):
    lib_merge_name = "merge_static_libs.bat"
    repo_ctx.template(
        lib_merge_name,
        Label("@onedal//dev/bazel/toolchains/tools:merge_static_libs_win.tpl.bat"),
        {"%{lib_path}": lib_path.replace("/", "\\")},
    )
    lib_merge_path = repo_ctx.path(lib_merge_name)
    return str(lib_merge_path)

def _create_dynamic_link_wrapper(repo_ctx, prefix, cc_path):
    wrapper_name = prefix + "_dynamic_link.bat"
    repo_ctx.template(
        wrapper_name,
        Label("@onedal//dev/bazel/toolchains/tools:dynamic_link_win.tpl.bat"),
        {"%{cc_path}": cc_path.replace("/", "\\")},
    )
    wrapper_path = repo_ctx.path(wrapper_name)
    return str(wrapper_path)

def _find_tools(repo_ctx, reqs):
    lib_path, _ = _find_tool(repo_ctx, "lib.exe", mandatory = True)
    cc_path, _ = _find_tool(repo_ctx, reqs.compiler_id, mandatory = True)
    strip_path, _ = _find_tool(repo_ctx, "dumpbin.exe", mandatory = True)
    dpcc_path, dpcpp_found = _find_tool(repo_ctx, reqs.dpc_compiler_id, mandatory = False)
    cc_link_path = _create_dynamic_link_wrapper(repo_ctx, "cc", cc_path)
    dpcc_link_path = _create_dynamic_link_wrapper(repo_ctx, "dpc", dpcc_path)
    if dpcpp_found:
        lib_path = cc_path[:-7] + "bin\\llvm-lib.exe"
    
    lib_merge_path = _create_lib_merge_tool(repo_ctx, lib_path)
    
    return struct(
        cc = cc_path,
        dpcc = dpcc_path,
        cc_link = cc_link_path,
        dpcc_link = dpcc_link_path,
        strip = strip_path,
        lib = lib_path,
        lib_merge = lib_merge_path,
        is_dpc_found = dpcpp_found,
        dpc_compiler_version = reqs.dpc_compiler_version
    )

def _prepare_builtin_include_directory_paths(repo_ctx, tools):
    required_tmp_includes = get_tmp_dpcpp_inc_directories(repo_ctx, tools) if tools.is_dpc_found else []
    builtin_include_directories = utils.unique(
        get_cxx_inc_directories(repo_ctx, tools.cc, "/TC") +
        get_cxx_inc_directories(repo_ctx, tools.cc, "/TP") +
        get_cxx_inc_directories(
            repo_ctx,
            tools.cc,
            "/TC",
            get_no_canonical_prefixes_opt(repo_ctx, tools.cc),
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.cc,
            "/TP",
            get_no_canonical_prefixes_opt(repo_ctx, tools.cc),
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "/TP",
            _add_msvc_toolchain_if_needed(repo_ctx, tools.dpcc),
        ) + 
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "/TP",
            get_no_canonical_prefixes_opt(repo_ctx, tools.dpcc) +
            _add_msvc_toolchain_if_needed(repo_ctx, tools.dpcc) +
            _add_sycl_linkage(repo_ctx, tools.dpcc) if tools.is_dpc_found else [],
        ) +
        required_tmp_includes,
    )
    write_builtin_include_directory_paths(repo_ctx, tools.cc, builtin_include_directories)
    return builtin_include_directories

def _get_bin_search_flag(repo_ctx, cc_path):
    cc_path = repo_ctx.path(cc_path)
    if not str(cc_path).startswith(str(repo_ctx.path(".")) + "\\"):
        bin_search_flag = ["/LIBPATH:" + str(cc_path.dirname)]
    else:
        bin_search_flag = []
    return bin_search_flag

def _get_msvc_toolchain_path(repo_ctx):
    cl_path = repo_ctx.which("cl.exe")
    if not cl_path:
        auto_configure_fail("Cannot find cl.exe; ensure MSVC is installed and in %PATH%")
    return str(cl_path.dirname.dirname.realpath)

def _add_msvc_toolchain_if_needed(repo_ctx, cc):
    if ("cl" in cc) or ("icx" in cc) or ("icpx" in cc):
        return ["/I" + _get_msvc_toolchain_path(repo_ctx) + "\\include"]
    return []

def _add_sycl_linkage(repo_ctx, cc):
    if ("icx" in cc) or ("icpx" in cc):
        return ["/Qsycl"]
    return []

def configure_cc_toolchain_win(repo_ctx, reqs):
    if reqs.os_id != "win":
        auto_configure_fail("Cannot configure Windows toolchain for '{}'".format(reqs.os_id))

    # Write empty C++ file to test compiler options
    repo_ctx.file(TEST_CPP_FILE, "int main() { return 0; }")

    # Default compilation/link options
    cxx_opts = ["/EHsc"]  # Enable C++ exception handling
    link_opts = ["/NOLOGO"]
    dynamic_link_libs = ["msvcrt.lib"]

    # Paths to tools/compiler includes
    tools = _find_tools(repo_ctx, reqs)
    builtin_include_directories = _prepare_builtin_include_directory_paths(repo_ctx, tools)

    # Addition compile/link flags
    bin_search_flag_cc = _get_bin_search_flag(repo_ctx, tools.cc)
    bin_search_flag_dpcc = _get_bin_search_flag(repo_ctx, tools.dpcc)

    # DPC++ kernel code split option
    dpcc_code_split = "per_kernel"

    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:cc_toolchain_win.tpl.BUILD"),
        {
            "%{cc_toolchain_identifier}": get_toolchain_identifier(reqs),
            "%{compiler}": reqs.compiler_id + "-" + reqs.compiler_version,
            "%{abi_version}": reqs.compiler_abi_version,
            "%{abi_libc_version}": reqs.libc_abi_version,
            "%{target_libc}": reqs.libc_version,
            "%{target_cpu}": reqs.target_arch_id,
            "%{host_system_name}": reqs.os_id + "-" + reqs.host_arch_id,
            "%{target_system_name}": reqs.os_id + "-" + reqs.target_arch_id,
            "%{supports_param_files}": "1",
            "%{compiler_deps}": get_starlark_list([
                ":builtin_include_directory_paths",
            ]),
            "%{ar_deps}": get_starlark_list([
                ":" + paths.basename(tools.ar_merge),
            ]),
            "%{linker_deps}": get_starlark_list([
                ":" + paths.basename(tools.cc_link),
                ":" + paths.basename(tools.dpcc_link),
            ]),

            "%{cc_path}": tools.cc.replace("/", "\\"),
            "%{dpcc_path}": tools.dpcc.replace("/", "\\"),
            "%{cc_link_path}": tools.cc_link.replace("/", "\\"),
            "%{dpcc_link_path}": tools.dpcc_link.replace("/", "\\"),
            "%{lib_path}": tools.lib.replace("/", "\\"),
            "%{lib_merge_path}": tools.lib_merge.replace("/", "\\"),
            "%{strip_path}": tools.strip.replace("/", "\\"),
            "%{cxx_builtin_include_directories}": get_starlark_list(builtin_include_directories),
            "%{compile_flags_cc}": get_starlark_list(
                _add_msvc_toolchain_if_needed(repo_ctx, tools.cc) +
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.cc,
                    is_dpcc = False,
                    category = "common",
                ) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "/diagnostics:caret",
                ),
            ),
            "%{compile_flags_dpcc}": get_starlark_list(
                _add_msvc_toolchain_if_needed(repo_ctx, tools.dpcc) +
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.dpcc,
                    is_dpcc = True,
                    category = "common",
                ) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "/Qsycl-device-code-split={}".format(dpcc_code_split),
                ) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "/diagnostics:caret",
                ),
            ) if tools.is_dpc_found else "",
            "%{compile_flags_pedantic_cc}": get_starlark_list(
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.cc,
                    is_dpcc = False,
                    category = "pedantic",
                ),
            ),
            "%{compile_flags_pedantic_dpcc}": get_starlark_list(
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.dpcc,
                    is_dpcc = True,
                    category = "pedantic",
                ),
            ) if tools.is_dpc_found else "",
            "%{cxx_flags}": get_starlark_list(cxx_opts),
            "%{link_flags_cc}": get_starlark_list(
                _add_msvc_toolchain_if_needed(repo_ctx, tools.cc) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "/DYNAMICBASE",
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "/NXCOMPAT",
                ) +
                bin_search_flag_cc + link_opts,
            ),
            "%{link_flags_dpcc}": get_starlark_list(
                _add_msvc_toolchain_if_needed(repo_ctx, tools.dpcc) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "/Qsycl",
                ) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "/Qsycl-device-code-split={}".format(dpcc_code_split),
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "/DYNAMICBASE",
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "/NXCOMPAT",
                ) +
                bin_search_flag_dpcc + link_opts,
            ) if tools.is_dpc_found else "",
            "%{dynamic_link_libs}": get_starlark_list(dynamic_link_libs),
            "%{opt_compile_flags}": get_starlark_list(
                [
                    "/Od",
                    "/Zi",
                    "/D_FORTIFY_SOURCE=2",
                ],
            ),

            "%{no_canonical_system_headers_flags_cc}": get_starlark_list(
                get_no_canonical_prefixes_opt(repo_ctx, tools.cc),
            ),
            "%{no_canonical_system_headers_flags_dpcc}": get_starlark_list(
                get_no_canonical_prefixes_opt(repo_ctx, tools.dpcc),
            ) if tools.is_dpc_found else "",
            "%{deterministic_compile_flags}": get_starlark_list(
                [
                    "/wd5105",
                    "/D__DATE__=\\\"redacted\\\"",
                    "/D__TIMESTAMP__=\\\"redacted\\\"",
                    "/D__TIME__=\\\"redacted\\\"",
                ],
            ),
            "%{dbg_compile_flags}": get_starlark_list(
                [
                    "/Zi",
                    "/DONEDAL_ENABLE_ASSERT",
                    "/Od",
                    "/DDEBUG_ASSERT",
                ],
            ),
            "%{supports_start_end_lib}": "False",
            "%{supports_random_seed}": "False",
            "%{cpu_flags_cc}": get_starlark_list_dict(
                get_cpu_specific_options(reqs),
            ),
            "%{cpu_flags_dpcc}": get_starlark_list_dict(
                get_cpu_specific_options(reqs, is_dpcc = True),
            ),
        },
    )
