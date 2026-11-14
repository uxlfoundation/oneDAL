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
    "split_escaped",
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
    "get_cpu_specific_options",
    "get_cross_tool_prefix",
    "get_cxx_inc_directories",
    "get_default_compiler_options",
    "get_no_canonical_prefixes_opt",
    "get_starlark_list_dict",
    "get_toolchain_identifier",
    "get_tmp_dpcpp_inc_directories",
)

def _find_tool(repo_ctx, tool_name, mandatory = False):
    if tool_name.startswith("/"):
        tool_path = repo_ctx.path(tool_name) if repo_ctx.path(tool_name).exists else None
    else:
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

def _find_tools(repo_ctx, reqs):
    # TODO: Use full compiler path from reqs
    # A cross-toolchain triple prefix (e.g. `aarch64-linux-gnu-`) applies to
    # binutils (`ar`, `strip`) as well as the compiler itself; without it,
    # cross-compiling with CC=aarch64-linux-gnu-gcc would silently pick up
    # the exec host's native `ar`/`strip`, producing wrong-arch archives.
    if reqs.downloaded_tools:
        ar_path = reqs.downloaded_tools.ar
        cc_path = reqs.downloaded_tools.cc
        strip_path = reqs.downloaded_tools.strip
    else:
        cross_prefix = get_cross_tool_prefix(repo_ctx)
        ar_path, _ = _find_tool(repo_ctx, cross_prefix + "ar", mandatory = True)
        # Use CC as given so that a versioned name (gcc-14, clang-19) or a full
        # path selects that exact compiler.
        cc_name = repo_ctx.os.environ.get("CC") or (cross_prefix + reqs.compiler_id)
        cc_path, _ = _find_tool(repo_ctx, cc_name, mandatory = True)
        strip_path, _ = _find_tool(repo_ctx, cross_prefix + "strip", mandatory = True)
    dpcc_path, dpcpp_found = _find_tool(repo_ctx, reqs.dpc_compiler_id, mandatory = False)
    if dpcpp_found:
        # The llvm-ar tool is used because Bazel prepends directory names with +,
        # which caused issues with GNU ar on RHEL. Derive it from the DPC++
        # compiler, not from the unrelated host C/C++ compiler.
        ar_path = paths.join(paths.dirname(dpcc_path), "compiler", "llvm-ar")
        if not repo_ctx.path(ar_path).exists:
            auto_configure_fail("Cannot find DPC++ archiver at {}".format(ar_path))

    return struct(
        cc = cc_path,
        dpcc = dpcc_path,
        strip = strip_path,
        ar = ar_path,
        is_dpc_found = dpcpp_found,
        dpc_compiler_version = reqs.dpc_compiler_version
    )

def _get_env_opts(repo_ctx, name):
    """The `:`-separated flags in environment variable `name`, as rules_cc reads them."""
    return [opt for opt in split_escaped(repo_ctx.os.environ.get(name, ""), ":") if opt]

def _escape_for_starlark_list(opts):
    """`opts` with quotes and backslashes escaped, since get_starlark_list inserts them verbatim."""
    return [repr(opt)[1:-1] for opt in opts]

def _preapre_builtin_include_directory_paths(repo_ctx, reqs, tools, cxx_opts):
    required_tmp_includes = get_tmp_dpcpp_inc_directories(repo_ctx, tools) if tools.is_dpc_found else []

    # Probe with the flags the compile actions pass: --gcc-toolchain can select another libstdc++,
    # and BAZEL_CXXOPTS can name another C++ library (-nostdinc++ -isystem <prefix>/include/c++/v1).
    cc_flags = _add_gcc_toolchain_if_needed(repo_ctx, tools.cc, reqs.compiler_id)
    cc_no_canonical_flags = cc_flags + get_no_canonical_prefixes_opt(repo_ctx, tools.cc)
    dpcc_flags = _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc, reqs.dpc_compiler_id)

    # As in rules_cc, the libc++ headers are builtin too, so a -stdlib=libc++ build works.
    libcxx_flags = add_compiler_option_if_supported(repo_ctx, tools.cc, "-stdlib=libc++")
    builtin_include_directories = utils.unique(
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc", cc_flags) +
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++", cc_flags + cxx_opts) +
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc", cc_no_canonical_flags) +
        get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++", cc_no_canonical_flags + cxx_opts) +
        (
            get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++", cc_flags + libcxx_flags + cxx_opts) +
            get_cxx_inc_directories(repo_ctx, tools.cc, "-xc++", cc_no_canonical_flags + libcxx_flags + cxx_opts) if libcxx_flags else []
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "-xc++",
            dpcc_flags + cxx_opts,
        ) +
        get_cxx_inc_directories(
            repo_ctx,
            tools.dpcc,
            "-xc++",
            get_no_canonical_prefixes_opt(repo_ctx, tools.dpcc) +
            dpcc_flags +
            _add_sycl_linkage(repo_ctx, tools.dpcc) + cxx_opts if tools.is_dpc_found else [],
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

def _get_selected_gcc_installation(repo_ctx, cc):
    """The GCC installation `cc` picks without --gcc-toolchain, or None."""
    marker = "Selected GCC installation: "
    for line in repo_ctx.execute([cc, "-v", "-E", "-x", "c++", "/dev/null"]).stderr.splitlines():
        if line.startswith(marker):
            return str(repo_ctx.path(line[len(marker):].strip()).realpath)
    return None

def _get_gcc_toolchain_path(repo_ctx, cc):
    # A gcc in PATH for another target (the host gcc next to a conda-forge clang) would pair
    # the compiler with a libstdc++ that does not match its sysroot, so prefer <triple>-gcc next to cc.
    gcc = repo_ctx.which("gcc")
    triple = repo_ctx.execute([cc, "-dumpmachine"]).stdout.strip()
    if not gcc or repo_ctx.execute([gcc, "-dumpmachine"]).stdout.strip() != triple:
        sibling_gcc = repo_ctx.path(cc).dirname.get_child(triple + "-gcc")
        if triple and sibling_gcc.exists:
            gcc = sibling_gcc
        elif gcc:
            # Keep the compiler's own GCC when it is built for exactly its triple and lies outside
            # the PATH gcc's prefix, as conda-forge clang's does; otherwise use the PATH gcc.
            selected = _get_selected_gcc_installation(repo_ctx, cc)
            if (selected and selected.split("/")[-2] == triple and
                not selected.startswith(str(gcc.dirname.dirname.realpath) + "/")):
                return None
    return str(gcc.dirname.dirname.realpath) if gcc else None

def _add_gcc_toolchain_if_needed(repo_ctx, cc, compiler_id):
    # By the detected id, not the name (CC=cc can be clang). icx is pinned to the same gcc as
    # icpx, as in Make; otherwise its default gcc lookup can find a different install.
    gcc_toolchain = _get_gcc_toolchain_path(repo_ctx, cc) if compiler_id in ["clang", "icx", "icpx"] else None
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
    # Probe --offload-compress with a shared link, as Make's dpcpp.mk:84-92 does.
    # A compile-only probe misses an icpx 2025.3 link failure; the probe runs in a fresh temp dir.
    tmp = repo_ctx.execute(["mktemp", "-d"])
    if tmp.return_code != 0:
        return []
    tmp_dir = tmp.stdout.strip()
    result = repo_ctx.execute([
        dpcc, "-fsycl", "--offload-new-driver", "--offload-compress",
        "-shared", "-fPIC", str(repo_ctx.path(TEST_CPP_FILE)),
        "-o", tmp_dir + "/probe.so",
    ], environment = {"TMPDIR": tmp_dir})
    repo_ctx.execute(["rm", "-rf", "--", tmp_dir])
    if result.return_code != 0:
        return []
    flags = ["--offload-new-driver"]

    # The legacy driver embeds -ftarget-register-alloc-mode=pvc:<opt> in the
    # device image, so the PVC-only JIT option is applied to PVC devices alone.
    # The new driver embeds no GRF option on its own, so pass it in that same
    # form. A bare -ze-intel-enable-auto-large-GRF-mode for every device breaks
    # the CPU OpenCL JIT ("Unrecognized build options").
    return flags + [
        "-Xsycl-target-backend=spir64",
        "-ftarget-register-alloc-mode=pvc:-ze-intel-enable-auto-large-GRF-mode",
    ]

def configure_cc_toolchain_lnx(repo_ctx, reqs):
    if reqs.os_id != "lnx":
        auto_configure_fail("Cannot configure Linux toolchain for '{}'".format(reqs.os_id))

    # Write empty C++ file to test compiler options
    repo_ctx.file(TEST_CPP_FILE, "int main() { return 0; }")

    # As in rules_cc's auto-configured toolchain, BAZEL_CXXOPTS adds flags to every C++ compile
    # and BAZEL_LINKOPTS to every link, for a compiler or C++ library outside the default paths.
    cxx_opts = _get_env_opts(repo_ctx, "BAZEL_CXXOPTS")
    link_opts = _get_env_opts(repo_ctx, "BAZEL_LINKOPTS")
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

    # Paths to tools/compiler includes
    tools = _find_tools(repo_ctx, reqs)
    builtin_include_directories = _preapre_builtin_include_directory_paths(repo_ctx, reqs, tools, cxx_opts)

    # Addition compile/link flags. Do not inject the host compiler directory
    # with -B for a GNU cross compiler: /usr/bin contains the host `ld`, which
    # overrides GCC's configured target linker (for example riscv64-linux-gnu-ld).
    cross_compiling = get_cross_tool_prefix(repo_ctx) != ""
    bin_search_flag_cc = [] if cross_compiling else _get_bin_search_flag(repo_ctx, tools.cc)
    bin_search_flag_dpcc = [] if cross_compiling else _get_bin_search_flag(repo_ctx, tools.dpcc)

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
            "%{host_cpu_constraint}": ARCH_ID_TO_PLATFORM_CPU[reqs.host_arch_id],
            "%{target_cpu_constraint}": ARCH_ID_TO_PLATFORM_CPU[reqs.target_arch_id],
            "%{supports_param_files}": "1",
            "%{compiler_deps}": get_starlark_list([
                ":builtin_include_directory_paths",
            ]),

            # Tools
            "%{cc_path}": tools.cc,
            "%{dpcc_path}": tools.dpcc,
            "%{ar_path}": tools.ar,
            "%{strip_path}": tools.strip,
            "%{cxx_builtin_include_directories}": get_starlark_list(builtin_include_directories),
            "%{compile_flags_cc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.cc, reqs.compiler_id) +
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
                _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc, reqs.dpc_compiler_id) +
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
            "%{cxx_flags}": get_starlark_list(_escape_for_starlark_list(cxx_opts)),
            "%{link_flags_cc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.cc, reqs.compiler_id) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-Wl,--as-needed",
                    "--as-needed",
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.cc,
                    "-Wl,-z,relro,-z,now",
                    "-z",
                ) +
                add_linker_option_if_supported(
                    # Matches the Make build's secure.opts.link.lnx; some linkers
                    # still default to a writable and executable stack.
                    repo_ctx,
                    tools.cc,
                    "-Wl,-z,noexecstack",
                    "-z",
                ) +
                add_compiler_option_if_supported(
                    # Have gcc return the exit code from ld.
                    repo_ctx,
                    tools.cc,
                    "-pass-exit-codes",
                ) +
                (
                    # `-static-intel` is not enough: the icx releases measured
                    # here still add `-limf` to a shared library link, so
                    # `libonedal.so` and `libonedal_core.so` end up recording
                    # DT_NEEDED `libimf.so`. Neither release stages the Intel
                    # compiler runtimes, so such a library fails to load on a
                    # host without oneAPI installed. The objects are already
                    # compiled with `-no-intel-lib` (dev/bazel/flags.bzl), so
                    # repeating it on the link line drops the dependency
                    # instead of satisfying it statically -- and it is what
                    # Make does: `link.dynamic.lnx.icx` in
                    # dev/make/compiler_definitions/icx.mkl.32e.mk passes
                    # `-no-intel-lib` and its libraries record no `libimf.so`.
                    ["-no-intel-lib", "-static-intel"] if reqs.compiler_id in ["icx", "icpx"] else []
                ) +
                bin_search_flag_cc + _escape_for_starlark_list(link_opts),
            ),
            "%{link_flags_dpcc}": get_starlark_list(
                _add_gcc_toolchain_if_needed(repo_ctx, tools.dpcc, reqs.dpc_compiler_id) +
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
                # Device link with the new offloading driver. The legacy driver
                # runs clang-offload-bundler, llvm-objcopy and
                # spirv-to-ir-wrapper once for every input object, including the
                # host-only DAAL objects that carry no device code. On
                # libonedal_dpc.so that is ~9.3k serial subprocesses. With this
                # flag the host-only objects are skipped and the DPC objects
                # keep their existing (legacy) format. Link actions only; compile
                # actions are unchanged.
                _add_new_offload_driver_link(repo_ctx, tools.dpcc) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "-Wl,--as-needed",
                    "--as-needed",
                ) +
                add_linker_option_if_supported(
                    repo_ctx,
                    tools.dpcc,
                    "-Wl,-z,relro,-z,now",
                    "-z",
                ) +
                add_linker_option_if_supported(
                    # Matches the Make build's secure.opts.link.lnx; some linkers
                    # still default to a writable and executable stack.
                    repo_ctx,
                    tools.dpcc,
                    "-Wl,-z,noexecstack",
                    "-z",
                ) +
                add_compiler_option_if_supported(
                    # Have gcc return the exit code from ld.
                    repo_ctx,
                    tools.dpcc,
                    "-pass-exit-codes",
                ) +
                bin_search_flag_dpcc + _escape_for_starlark_list(link_opts),
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

                    # Disable loop transformation warnings that appear with -O1
                    # and are treated as errors in dbg configuration
                    "-Wno-pass-failed",

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
