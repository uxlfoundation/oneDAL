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

load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
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
    if tool_name.startswith("/"):
        return tool_name
    tool_path = repo_ctx.which(tool_name)
    is_found = tool_path != None
    if not is_found:
        if mandatory:
            auto_configure_fail("Cannot find {}; try to correct your $PATH".format(tool_name))
        else:
            repo_ctx.template(
                "tool_not_found.sh",
                Label("@onedal//dev/bazel/toolchains/tools:tool_not_found.tpl.sh"),
                {"%{tool_name}": tool_name},
            )
            tool_path = repo_ctx.path("tool_not_found.sh")
    return str(tool_path), is_found

def find_tool(repo_ctx, tool_name, mandatory = False):
    return _find_tool(repo_ctx, tool_name, mandatory)

def _create_ar_merge_tool(repo_ctx, ar_path):
    ar_merge_name = "merge_static_libs.sh"
    repo_ctx.template(
        ar_merge_name,
        Label("@onedal//dev/bazel/toolchains/tools:merge_static_libs_lnx.tpl.sh"),
        {"%{ar_path}": ar_path},
    )
    ar_merge_path = repo_ctx.path(ar_merge_name)
    return str(ar_merge_path)

def _create_dynamic_link_wrapper(repo_ctx, prefix, cc_path):
    wrapper_name = prefix + "_dynamic_link.sh"
    repo_ctx.template(
        wrapper_name,
        Label("@onedal//dev/bazel/toolchains/tools:dynamic_link_lnx.tpl.sh"),
        {"%{cc_path}": cc_path},
    )
    wrapper_path = repo_ctx.path(wrapper_name)
    return str(wrapper_path)

def _find_tools(repo_ctx, reqs):
    # TODO: Use full compiler path from reqs
    ar_path, _ = _find_tool(repo_ctx, "ar", mandatory = True)
    cc_path, _ = _find_tool(repo_ctx, reqs.compiler_id, mandatory = True)
    strip_path, _ = _find_tool(repo_ctx, "strip", mandatory = True)
    dpcc_path, dpcpp_found = _find_tool(repo_ctx, reqs.dpc_compiler_id, mandatory = False)
    cc_link_path = _create_dynamic_link_wrapper(repo_ctx, "cc", cc_path)
    dpcc_link_path = _create_dynamic_link_wrapper(repo_ctx, "dpc", dpcc_path)
    ar_merge_path = _create_ar_merge_tool(repo_ctx, ar_path)
    return struct(
        cc = cc_path,
        dpcc = dpcc_path,
        cc_link = cc_link_path,
        dpcc_link = dpcc_link_path,
        strip = strip_path,
        ar = ar_path,
        ar_merge = ar_merge_path,
        is_dpc_found = dpcpp_found,
        dpc_compiler_version = reqs.dpc_compiler_version
    )

def _preapre_builtin_include_directory_paths(repo_ctx, tools):
    required_tmp_includes = get_tmp_dpcpp_inc_directories(repo_ctx, tools) if tools.is_dpc_found else []
    builtin_include_directories = utils.unique(
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc") +
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++") +
        get_cxx_inc_directories(
            repo_ctx,
            tools.cc,
            "-xc",
            get_no_canonical_prefixes_opt(repo_ctx, tools.cc),
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.cc,
            "-xc++",
            get_no_canonical_prefixes_opt(repo_ctx, tools.cc),
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "-xc++",
            _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc),
        ) + 
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "-xc++",
            get_no_canonical_prefixes_opt(repo_ctx, tools.dpcc) +
            _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc) +
            _add_sycl_linkage(repo_ctx, tools.dpcc) if tools.is_dpc_found else [],
        ) +
        required_tmp_includes,
    )
    write_builtin_include_directory_paths(repo_ctx, tools.cc, builtin_include_directories)
    return builtin_include_directories

def _get_bin_search_flag(repo_ctx, cc_path):
    cc_path = repo_ctx.path(cc_path)
    if not str(cc_path).startswith(str(repo_ctx.path(".")) + "/"):
        # cc is outside the repository, set -B
        bin_search_flag = ["-B" + str(cc_path.dirname)]
    else:
        # cc is inside the repository, don't set -B.
        bin_search_flag = []
    return bin_search_flag

def _get_gcc_toolchain_path(repo_ctx):
    return str(repo_ctx.which("gcc").dirname.dirname.realpath)

def _add_gcc_toolchain_if_needed(repo_ctx, cc):
    if ("clang" in cc) or ("icpx" in cc):
        return ["--gcc-toolchain=" + _get_gcc_toolchain_path(repo_ctx)]
    else:
        return []

def _add_sycl_linkage(repo_ctx, cc):
    if ("icx" in cc) or ("icpx" in cc):
        return ["-fsycl"]
    else:
        return []

def configure_cc_toolchain_lnx(repo_ctx, reqs):
    if reqs.os_id != "lnx":
        auto_configure_fail("Cannot configure Linux toolchain for '{}'".format(reqs.os_id))

    # Write empty C++ file to test compiler options
    repo_ctx.file(TEST_CPP_FILE, "int main() { return 0; }")

    # Default compilations/link options
    cxx_opts = []
    link_opts = []
    dynamic_link_libs = ["-lstdc++", "-lm", "-ldl"]

    # Paths to tools/compiler includes
    tools = _find_tools(repo_ctx, reqs)
    builtin_include_directories = _preapre_builtin_include_directory_paths(repo_ctx, tools)

    # Addition compile/link flags
    bin_search_flag_cc = _get_bin_search_flag(repo_ctx, tools.cc)
    bin_search_flag_dpcc = _get_bin_search_flag(repo_ctx, tools.dpcc)

    # DPC++ kernel code split option
    dpcc_code_split = "per_kernel"

    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:cc_toolchain_lnx.tpl.BUILD"),
        {
            # Various IDs
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

            # Tools
            "%{cc_path}": tools.cc,
            "%{dpcc_path}": tools.dpcc,
            "%{cc_link_path}": tools.cc_link,
            "%{dpcc_link_path}": tools.dpcc_link,
            "%{ar_path}": tools.ar,
            "%{ar_merge_path}": tools.ar_merge,
            "%{strip_path}": tools.strip,
            "%{cxx_builtin_include_directories}": get_starlark_list(builtin_include_directories),
            "%{compile_flags_cc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.cc) +
                get_default_compiler_options(
                    repo_ctx,
                    reqs,
                    tools.cc,
                    is_dpcc = False,
                    category = "common",
                ) +
                add_compiler_option_if_supported(
                    # Option supported only by Intel Compiler to disable some warnings:
                    # https://software.intel.com/comment/1848937
                    repo_ctx,
                    tools.cc,
                    "-diag-disable=remark",
                ) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-fdiagnostics-color=always",
                ),
            ),
            "%{compile_flags_dpcc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc) +
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
                    "-fsycl-device-code-split={}".format(dpcc_code_split),
                ) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-fdiagnostics-color=always",
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
                _add_gcc_toolchain_if_needed(repo_ctx, tools.cc) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-Wl,-no-as-needed",
                    "-no-as-needed",
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-Wl,-z,relro,-z,now",
                    "-z",
                ) +
                add_compiler_option_if_supported(
                    # Have gcc return the exit code from ld.
                    repo_ctx,
                    tools.cc,
                    "-pass-exit-codes",
                ) +
                (
                    ["-no-cilk", "-static-intel"] if reqs.compiler_id == "icc" else []
                ) +
                (
                    ["-static-intel"] if reqs.compiler_id in ["icx", "icpx"] else []
                ) +
                bin_search_flag_cc + link_opts,
            ),
            "%{link_flags_dpcc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "-fsycl",
                ) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "-fsycl-device-code-split={}".format(dpcc_code_split),
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "-Wl,-no-as-needed",
                    "-no-as-needed",
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "-Wl,-z,relro,-z,now",
                    "-z",
                ) +
                add_compiler_option_if_supported(
                    # Have gcc return the exit code from ld.
                    repo_ctx,
                    tools.dpcc,
                    "-pass-exit-codes",
                ) +
                bin_search_flag_dpcc + link_opts,
            ) if tools.is_dpc_found else "",
            "%{dynamic_link_libs}": get_starlark_list(dynamic_link_libs),
            "%{opt_compile_flags}": get_starlark_list(
                [
                    # No debug symbols.
                    "-g0",

                    # Conservative choice for -O
                    "-O2",

                    # It turns out that some GCC builds set _FORTIFY_SOURCE internally,
                    # so we need to undefine it first
                    "-U_FORTIFY_SOURCE",

                    # Security hardening on by default.
                    "-D_FORTIFY_SOURCE=2",

                    # Removal of unused code and data at link time (can this increase binary
                    # size in some cases?).
                    "-ffunction-sections",
                    "-fdata-sections",
                ],
            ),
            "%{opt_link_flags}": get_starlark_list(
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-Wl,--gc-sections",
                    "-gc-sections",
                ),
            ),
            "%{no_canonical_system_headers_flags_cc}": get_starlark_list(
                # Probably bug: Intel Compiler links OpenMP runtime if
                # -fno-canonical-system-headers is provided
                (get_no_canonical_prefixes_opt(repo_ctx, tools.cc) if reqs.compiler_id not in ["icx", "icpx"] else []),
            ),
            "%{no_canonical_system_headers_flags_dpcc}": get_starlark_list(
                get_no_canonical_prefixes_opt(repo_ctx, tools.dpcc),
            ) if tools.is_dpc_found else "",
            "%{deterministic_compile_flags}": get_starlark_list(
                [
                    # Make C++ compilation deterministic. Use linkstamping instead of these
                    # compiler symbols.
                    "-Wno-builtin-macro-redefined",
                    "-D__DATE__=\\\"redacted\\\"",
                    "-D__TIMESTAMP__=\\\"redacted\\\"",
                    "-D__TIME__=\\\"redacted\\\"",
                ],
            ),
            "%{dbg_compile_flags}": get_starlark_list(
                [
                    "-g",

                    # Enable assertions
                    "-DONEDAL_ENABLE_ASSERT",

                    # Disable optimizations explicitly
                    # Some compilers like Intel uses -O2 by default
                    # Prefer -O1 because -O0 code may be painfully slow
                    "-O1",

                    # oneDAL specific defined to enabled assertions
                    "-DDEBUG_ASSERT",
                ],
            ),
            "%{supports_start_end_lib}": "False" if reqs.compiler_id in ["icx", "icpx"] else "True",
            "%{supports_random_seed}": "False" if reqs.compiler_id in ["icx", "icpx"] else "True",
            "%{cpu_flags_cc}": get_starlark_list_dict(
                get_cpu_specific_options(reqs),
            ),
            "%{cpu_flags_dpcc}": get_starlark_list_dict(
                get_cpu_specific_options(reqs, is_dpcc = True),
            ),
        },
    )
