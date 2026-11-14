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
    "ARCH_ID_TO_PLATFORM_CPU",
    "TEST_CPP_FILE",
    "add_compiler_option_if_supported",
    "add_linker_option_if_supported",
    "find_tool",
    "get_cpu_specific_options",
    "get_cross_tool_prefix",
    "get_cxx_inc_directories",
    "get_default_compiler_options",
    "get_no_canonical_prefixes_opt",
    "get_starlark_list_dict",
    "get_toolchain_identifier",
)

def _find_tools(repo_ctx, reqs):
    # TODO: Use full compiler path from reqs
    # The cross triple prefix applies to ar and strip too, or a cross build gets the host's.
    if reqs.downloaded_tools:
        ar_path = reqs.downloaded_tools.ar
        cc_path = reqs.downloaded_tools.cc
        strip_path = reqs.downloaded_tools.strip
    else:
        cross_prefix = get_cross_tool_prefix(repo_ctx)
        ar_path, _ = find_tool(repo_ctx, cross_prefix + "ar", mandatory = True)
        # Use CC as given so that a versioned name (gcc-14, clang-19) or a full
        # path selects that exact compiler.
        cc_name = repo_ctx.os.environ.get("CC") or (cross_prefix + reqs.compiler_id)
        cc_path, _ = find_tool(repo_ctx, cc_name, mandatory = True)
        strip_path, _ = find_tool(repo_ctx, cross_prefix + "strip", mandatory = True)
    dpcc_path, dpcpp_found = find_tool(repo_ctx, reqs.dpc_compiler_id, mandatory = False)
    if dpcpp_found:
        # GNU ar on RHEL mishandles the `+` Bazel puts in directory names.
        ar_path = paths.join(paths.dirname(dpcc_path), "compiler", "llvm-ar")
        if not repo_ctx.path(ar_path).exists:
            auto_configure_fail("Cannot find DPC++ archiver at {}".format(ar_path))

    return struct(
        cc = cc_path,
        dpcc = dpcc_path,
        strip = strip_path,
        ar = ar_path,
        is_dpc_found = dpcpp_found,
    )

def _prepare_builtin_include_directory_paths(repo_ctx, tools, cc_probe, dpcc_probe):
    dpcc_include_directories = (
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "-xc++",
            dpcc_probe.gcc_toolchain,
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "-xc++",
            dpcc_probe.no_canonical + dpcc_probe.gcc_toolchain + _add_sycl_linkage(repo_ctx, tools.dpcc),
        ) +
        # icpx includes the integration header it generates under /tmp, which Bazel
        # then requires to be a builtin include directory.
        ["/tmp"]
    ) if tools.is_dpc_found else []

    # Probe with the flags the compile actions pass: --gcc-toolchain can select another libstdc++.
    cc_flags = cc_probe.gcc_toolchain
    cc_no_canonical_flags = cc_flags + cc_probe.no_canonical

    # As in rules_cc, the libc++ headers are builtin too, so a -stdlib=libc++ build works.
    libcxx_flags = add_compiler_option_if_supported(repo_ctx, tools.cc, "-stdlib=libc++")
    builtin_include_directories = utils.unique(
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc", cc_flags) +
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++", cc_flags) +
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc", cc_no_canonical_flags) +
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++", cc_no_canonical_flags) +
        (
            get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++", cc_flags + libcxx_flags) +
            get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++", cc_no_canonical_flags + libcxx_flags) if libcxx_flags else []
        ) +
        dpcc_include_directories,
    )
    write_builtin_include_directory_paths(repo_ctx, tools.cc, builtin_include_directories)
    return builtin_include_directories

def _get_bin_search_flag(repo_ctx, cc_path):
    cc_path = repo_ctx.path(cc_path)
    if not str(cc_path).startswith(str(repo_ctx.path(".")) + "/"):
        bin_search_flag = ["-B" + str(cc_path.dirname)]
    else:
        bin_search_flag = []
    return bin_search_flag

def _get_gcc_toolchain_path(repo_ctx, cc):
    # A gcc in PATH for another target (the host gcc next to a conda-forge clang) would pair
    # the compiler with a libstdc++ that does not match its sysroot, so prefer <triple>-gcc next to cc.
    gcc = repo_ctx.which("gcc")
    triple = repo_ctx.execute([cc, "-dumpmachine"]).stdout.strip()
    if not gcc or repo_ctx.execute([gcc, "-dumpmachine"]).stdout.strip() != triple:
        sibling_gcc = repo_ctx.path(cc).dirname.get_child(triple + "-gcc")
        if triple and sibling_gcc.exists:
            gcc = sibling_gcc
    return str(gcc.dirname.dirname.realpath) if gcc else None

def _add_gcc_toolchain_if_needed(repo_ctx, cc):
    # Pin icx to the same gcc as icpx (the one on PATH), as Make does. Without it, icx's cc
    # actions fall back to icx's default gcc lookup, which can be a different install.
    gcc_toolchain = _get_gcc_toolchain_path(repo_ctx, cc) if ("clang" in cc) or ("icx" in cc) or ("icpx" in cc) else None
    if gcc_toolchain:
        return ["--gcc-toolchain=" + gcc_toolchain]
    else:
        # Without a gcc driver in PATH, clang finds the installed libstdc++ itself.
        return []

def _add_sycl_linkage(repo_ctx, cc):
    if ("icx" in cc) or ("icpx" in cc):
        return ["-fsycl"]
    else:
        return []

def _add_new_offload_driver_link(repo_ctx, dpcc):
    # Probe --offload-compress with a shared link, as Make's dpcpp.mk does: a compile-only
    # probe misses an icpx 2025.3 link failure. The probe gets a scratch dir of its own.
    scratch_dir = repo_ctx.path("offload_probe")
    repo_ctx.file("offload_probe/.keep", "")
    result = repo_ctx.execute([
        dpcc, "-fsycl", "--offload-new-driver", "--offload-compress",
        "-shared", "-fPIC", str(repo_ctx.path(TEST_CPP_FILE)),
        "-o", str(scratch_dir.get_child("probe.so")),
    ], environment = {"TMPDIR": str(scratch_dir)})
    repo_ctx.delete(scratch_dir)
    if result.return_code != 0:
        return []
    flags = ["--offload-new-driver"]

    # The GRF option the legacy driver embeds, for PVC only: on every device it breaks the
    # CPU OpenCL JIT ("Unrecognized build options").
    return flags + [
        "-Xsycl-target-backend=spir64",
        "-ftarget-register-alloc-mode=pvc:-ze-intel-enable-auto-large-GRF-mode",
    ]

def _secure_link_flags(repo_ctx, driver):
    # Make's secure.opts.link.lnx; some linkers still default to a writable and executable stack.
    # -pass-exit-codes has gcc return the exit code from ld.
    return (
        add_linker_option_if_supported(repo_ctx, driver, "-Wl,--as-needed", "--as-needed") +
        add_linker_option_if_supported(repo_ctx, driver, "-Wl,-z,relro,-z,now", "-z") +
        add_linker_option_if_supported(repo_ctx, driver, "-Wl,-z,noexecstack", "-z") +
        add_compiler_option_if_supported(repo_ctx, driver, "-pass-exit-codes")
    )

def _probe_driver(repo_ctx, driver):
    """Flags probed once per compiler driver and shared by its compile, link and include setup."""
    return struct(
        gcc_toolchain = _add_gcc_toolchain_if_needed(repo_ctx, driver),
        no_canonical = get_no_canonical_prefixes_opt(repo_ctx, driver),
        secure_link = _secure_link_flags(repo_ctx, driver),
    )

def configure_cc_toolchain_lnx(repo_ctx, reqs):
    if reqs.os_id != "lnx":
        auto_configure_fail("Cannot configure Linux toolchain for '{}'".format(reqs.os_id))

    repo_ctx.file(TEST_CPP_FILE, "int main() { return 0; }")

    cxx_opts = []
    link_opts = []
    # -lpthread -ldl stay outside --as-needed, as in Make (lnx32e.mk, deps.mkl.mk).
    # They add NEEDED entries on glibc < 2.34 and resolve to empty stubs after it.
    dynamic_link_libs = [
        "-lstdc++",
        "-lm",
        "-Wl,--no-as-needed",
        "-lpthread",
        "-ldl",
        "-Wl,--as-needed",
    ]

    tools = _find_tools(repo_ctx, reqs)
    cc_probe = _probe_driver(repo_ctx, tools.cc)
    dpcc_probe = _probe_driver(repo_ctx, tools.dpcc) if tools.is_dpc_found else None
    builtin_include_directories = _prepare_builtin_include_directory_paths(repo_ctx, tools, cc_probe, dpcc_probe)

    # No -B for a GNU cross compiler: /usr/bin holds the host `ld`, which would override its own.
    cross_compiling = get_cross_tool_prefix(repo_ctx) != ""
    bin_search_flag_cc = [] if cross_compiling else _get_bin_search_flag(repo_ctx, tools.cc)
    bin_search_flag_dpcc = [] if cross_compiling or not tools.is_dpc_found else _get_bin_search_flag(repo_ctx, tools.dpcc)

    # DPC++ kernel code split option
    dpcc_code_split = "per_kernel"

    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:cc_toolchain_lnx.tpl.BUILD"),
        {
            "%{cc_toolchain_identifier}": get_toolchain_identifier(reqs),
            "%{compiler}": reqs.compiler_id + "-" + reqs.compiler_version,
            "%{abi_version}": reqs.compiler_abi_version,
            "%{abi_libc_version}": reqs.libc_abi_version,
            "%{target_libc}": reqs.libc_version,
            "%{target_cpu}": reqs.target_arch_id,
            "%{host_system_name}": reqs.os_id + "-" + reqs.host_arch_id,
            "%{target_system_name}": reqs.os_id + "-" + reqs.target_arch_id,
            "%{host_cpu_constraint}": ARCH_ID_TO_PLATFORM_CPU[reqs.host_arch_id],
            "%{target_cpu_constraint}": ARCH_ID_TO_PLATFORM_CPU[reqs.target_arch_id],
            "%{supports_param_files}": "1",
            "%{compiler_deps}": get_starlark_list([
                ":builtin_include_directory_paths",
            ]),

            "%{cc_path}": tools.cc,
            "%{dpcc_path}": tools.dpcc,
            "%{ar_path}": tools.ar,
            "%{strip_path}": tools.strip,
            "%{cxx_builtin_include_directories}": get_starlark_list(builtin_include_directories),
            "%{compile_flags_cc}": get_starlark_list(
                cc_probe.gcc_toolchain +
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
                    "-diag-disable=remark",
                ) +
                add_compiler_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-fdiagnostics-color=always",
                ),
            ),
            "%{compile_flags_dpcc}": get_starlark_list(
                dpcc_probe.gcc_toolchain +
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
                    tools.dpcc,
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
                cc_probe.gcc_toolchain +
                cc_probe.secure_link +
                (
                    # icx adds -limf to shared links despite -static-intel, and the release ships no
                    # libimf.so; Make: link.dynamic.lnx.icx.
                    ["-no-intel-lib", "-static-intel"] if reqs.compiler_id in ["icx", "icpx"] else []
                ) +
                bin_search_flag_cc + link_opts,
            ),
            "%{link_flags_dpcc}": get_starlark_list(
                dpcc_probe.gcc_toolchain +
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
                # The legacy offload driver runs three tools per input object, host-only DAAL
                # objects included; the new one skips those.
                _add_new_offload_driver_link(repo_ctx, tools.dpcc) +
                dpcc_probe.secure_link +
                bin_search_flag_dpcc + link_opts,
            ) if tools.is_dpc_found else "",
            "%{dynamic_link_libs}": get_starlark_list(dynamic_link_libs),
            "%{opt_compile_flags}": get_starlark_list(
                [
                    "-g0",

                    "-O2",

                    # Some GCC builds predefine _FORTIFY_SOURCE.
                    "-U_FORTIFY_SOURCE",

                    "-D_FORTIFY_SOURCE=2",

                    # For --gc-sections.
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
                # icx links the OpenMP runtime when given -fno-canonical-system-headers.
                (cc_probe.no_canonical if reqs.compiler_id not in ["icx", "icpx"] else []),
            ),
            "%{no_canonical_system_headers_flags_dpcc}": get_starlark_list(
                dpcc_probe.no_canonical,
            ) if tools.is_dpc_found else "",
            "%{deterministic_compile_flags}": get_starlark_list(
                [
                    "-Wno-builtin-macro-redefined",
                    "-D__DATE__=\\\"redacted\\\"",
                    "-D__TIMESTAMP__=\\\"redacted\\\"",
                    "-D__TIME__=\\\"redacted\\\"",
                ],
            ),
            "%{dbg_compile_flags}": get_starlark_list(
                [
                    "-g",

                    "-DONEDAL_ENABLE_ASSERT",

                    # Explicit, as icx defaults to -O2; -O0 code is too slow.
                    "-O1",

                    # Loop transformation warnings appear at -O1.
                    "-Wno-pass-failed",

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
