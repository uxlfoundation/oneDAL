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

def _create_dpc_link_wrapper(repo_ctx, dpcc_path):
    repo_ctx.template(
        "dpc_link_win.ps1",
        Label("@onedal//dev/bazel/toolchains/tools:dpc_link_win.ps1"),
        {},
    )
    repo_ctx.template(
        "dpc_link_win.bat",
        Label("@onedal//dev/bazel/toolchains/tools:dpc_link_win.tpl.bat"),
        {"%{dpcc}": dpcc_path},
    )
    return str(repo_ctx.path("dpc_link_win.bat"))

def _find_tools_icx(repo_ctx):
    # Use `icx.exe` (clang-cl driver) for C and C++/DPC++ compile.
    # For host link, call `lld-link.exe` (or link.exe) DIRECTLY rather
    # than going through the icx driver — mirrors makefile common.mk:125
    # (`link.dynamic.win = link ...`) and Linux's dynamic_link_lnx.tpl.sh
    # wrapper. Going through icx for huge link actions caused LNK1170:
    # icx materialised every Bazel-line arg into a single >131071-char
    # line in its own intermediate response file before invoking link.exe.
    cc_path, _ = _find_tool(repo_ctx, "icx", mandatory = True)
    dpcc_path, dpcpp_found = _find_tool(repo_ctx, "icx", mandatory = False)
    dpcc_link_path = _create_dpc_link_wrapper(repo_ctx, dpcc_path) if dpcpp_found else dpcc_path
    # Host link tool: prefer lld-link.exe (ships next to icx in oneAPI),
    # fall back to MSVC's link.exe — both accept the same flag syntax and
    # multi-line response files.
    cc_link_path, lld_found = _find_tool(repo_ctx, "lld-link", mandatory = False)
    if not lld_found:
        cc_link_path, _ = _find_tool(repo_ctx, "link", mandatory = True)
    # Static archiver: same as before, llvm-lib first then MSVC lib.
    ar_path, _ = _find_tool(repo_ctx, "llvm-lib", mandatory = False)
    if not ar_path or ar_path.endswith("tool_not_found_llvm-lib.bat"):
        ar_path, _ = _find_tool(repo_ctx, "lib", mandatory = True)
    return struct(
        cc = cc_path,
        dpcc = dpcc_path,
        # Host link: lld-link.exe directly. DPC++ link still goes through
        # icx so it can pull in SYCL device-code libs (matches the
        # makefile's dpc.link.dynamic.win path in common.mk:138).
        cc_link = cc_link_path,
        dpcc_link = dpcc_link_path,
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

    # Collect system include roots. The INCLUDE env (populated by Intel
    # setvars.bat + MSVC VsDevCmd.bat) covers MSVC, Windows SDK, and oneAPI
    # headers. Add the clang builtin include dir shipped in the oneAPI
    # compiler package — it is not exported via INCLUDE but icx pulls
    # headers from there for its intrinsic declarations and the C11
    # `__stddef_*` helpers. Mirrors how the Makefile relies on the oneAPI
    # install layout without probing the compiler.
    builtin_include_directories = []

    def _add_builtin_include_dir(path):
        p = path.strip().replace("\\", "/")
        if p and p not in builtin_include_directories:
            builtin_include_directories.append(p)
        if p:
            real = str(repo_ctx.path(p).realpath).replace("\\", "/")
            if real and real not in builtin_include_directories:
                builtin_include_directories.append(real)

    for raw in repo_ctx.os.environ.get("INCLUDE", "").split(";"):
        _add_builtin_include_dir(raw)
    cmplr_root = repo_ctx.os.environ.get("CMPLR_ROOT", "").strip().replace("\\", "/")
    if cmplr_root:
        # Whitelist the compiler package root as well as the precise include
        # dirs. oneAPI's `latest` can be a junction/symlink while icx reports
        # absolute inclusions through the resolved versioned path
        # (`.../compiler/2026.0/...`), and Bazel compares the resolved header
        # path against this list.
        _add_builtin_include_dir(cmplr_root)
        # oneAPI's math/intrinsic overrides ship here and are pulled in by
        # icx before the system headers — Bazel sees them as absolute-path
        # inclusions and needs them whitelisted as toolchain dirs.
        opt_inc = "{}/opt/compiler/include".format(cmplr_root)
        if repo_ctx.path(opt_inc).exists:
            _add_builtin_include_dir(opt_inc)
        # Clang builtin headers live under `lib/clang/<N>/include`.
        clang_root = repo_ctx.path("{}/lib/clang".format(cmplr_root))
        if clang_root.exists:
            _add_builtin_include_dir(str(clang_root))
            for entry in clang_root.readdir():
                inc = "{}/include".format(str(entry).replace("\\", "/"))
                if repo_ctx.path(inc).exists:
                    _add_builtin_include_dir(inc)

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

    # `--driver-mode=g++` is not added to the flag sets — the `icx_g_driver.bat`
    # wrapper registered in `_find_tools_icx` injects it before any bazel-
    # supplied argument (including those read from @paramfile, where flag-set
    # flags would arrive too late for clang-cl to re-parse).
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

    # icx in clang-cl mode picks up lld-link (or link.exe) automatically.
    # lld-link resolves `<name>.lib` via the `LIB` env var; Bazel actions
    # do not inherit the shell `LIB`, so pass it through via the toolchain
    # env_entries (see cc_toolchain_config_win.bzl — `msvc_env_feature`).
    link_flags_cc = []
    link_flags_dpcc = ([
        "-fsycl",
        "-fsycl-device-code-split={}".format(dpcc_code_split),
        # The icx driver does not add the SYCL runtime import library for
        # Bazel's DLL link path when most device objects are supplied through
        # a /WHOLEARCHIVE static library. Link it explicitly so generated
        # registration thunks resolve __sycl_{,un}register_lib on Windows.
        "sycl.lib",
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
                # clang-cl: `/Z7` embeds debug info in the .obj (no .pdb
                # juggling, matches Makefile `-DEBC.icx = -debug:all -Z7`).
                "/Z7",
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
            # BUILD template values are embedded in double-quoted Starlark
            # strings, so any literal backslash becomes an invalid escape.
            # Convert to forward slashes (both lld-link and MSVC link.exe
            # accept forward-slash paths in INCLUDE/LIB/PATH entries).
            "%{env_include}": repo_ctx.os.environ.get("INCLUDE", "").replace("\\", "/"),
            "%{env_lib}": repo_ctx.os.environ.get("LIB", "").replace("\\", "/"),
            "%{env_path}": repo_ctx.os.environ.get("PATH", "").replace("\\", "/"),
        },
    )

def configure_cc_toolchain_win(repo_ctx, reqs):
    if _is_icx_requested(repo_ctx):
        _configure_cc_toolchain_win_icx(repo_ctx, reqs)
    else:
        _configure_cc_toolchain_win_cl(repo_ctx)
