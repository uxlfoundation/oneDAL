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

load("@rules_cc//cc/private/toolchain:windows_cc_configure.bzl", "configure_windows_toolchain")
load("@rules_cc//cc/private/toolchain:lib_cc_configure.bzl",
    "get_starlark_list",
)
load("@onedal//dev/bazel:utils.bzl", "paths")
load("@onedal//dev/bazel/toolchains:common.bzl",
    "ARCH_ID_TO_PLATFORM_CPU",
    "TEST_CPP_FILE",
    "add_compiler_option_if_supported",
    "get_cpu_specific_options",
    "get_default_compiler_options",
    "get_starlark_list_dict",
    "get_toolchain_identifier",
)

def _find_tool(repo_ctx, tool_name, mandatory = False):
    tool_path = repo_ctx.which(tool_name)
    if tool_path == None:
        if mandatory:
            fail("Cannot find {}; is it on PATH?".format(tool_name))
        return "", False
    return str(tool_path), True

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

def _find_llvm_tool(repo_ctx, cc_path, name, fallback):
    """Finds an LLVM tool of the icx toolchain, or the MSVC tool it replaces.

    oneAPI ships its LLVM tools in `compiler/`, a subdirectory of the icx
    directory that is not on PATH. The MSVC fallbacks are not long-path aware.

    @param cc_path   Path of icx.
    @param name      LLVM tool, such as `lld-link`.
    @param fallback  MSVC tool to use when `name` is not found, such as `link`.
    @return          Path of the tool to use.
    """
    path, found = _find_tool(repo_ctx, name, mandatory = False)
    if found:
        return path
    bundled = repo_ctx.path(cc_path).dirname.get_child("compiler").get_child(name + ".exe")
    if bundled.exists:
        return str(bundled)
    print("{} not found; falling back to MSVC {}.exe, which is not long-path aware".format(
        name, fallback))
    path, _ = _find_tool(repo_ctx, fallback, mandatory = True)
    return path

def _find_tools_icx(repo_ctx):
    # Host links run the linker directly, as Make's `link.dynamic.win`: icx writes all arguments
    # of a large link into one response-file line and fails with LNK1170.
    cc_path, _ = _find_tool(repo_ctx, "icx", mandatory = True)
    dpcc_link_path = _create_dpc_link_wrapper(repo_ctx, cc_path)
    cc_link_path = _find_llvm_tool(repo_ctx, cc_path, "lld-link", "link")
    ar_path = _find_llvm_tool(repo_ctx, cc_path, "llvm-lib", "lib")
    return struct(
        cc = cc_path,
        dpcc = cc_path,
        # DPC++ links go through icx for the SYCL device libraries (Make: dpc.link.dynamic.win).
        cc_link = cc_link_path,
        dpcc_link = dpcc_link_path,
        ar = ar_path,
        is_dpc_found = True,
    )

def _find_tools_clang(repo_ctx):
    """Upstream LLVM clang-cl tools, as Make's `PLAT=winarm`; host only, as icpx targets x86_64 only."""
    cc_path, _ = _find_tool(repo_ctx, "clang-cl", mandatory = True)
    # Required, as in Make: falling back to `link` could find MSYS2's unrelated link.exe.
    cc_link_path, _ = _find_tool(repo_ctx, "lld-link", mandatory = True)
    ar_path, _ = _find_tool(repo_ctx, "llvm-lib", mandatory = True)
    return struct(
        cc = cc_path,
        dpcc = "",
        cc_link = cc_link_path,
        dpcc_link = "",
        ar = ar_path,
        is_dpc_found = False,
    )

def _requested_compiler(repo_ctx):
    """Which Windows compiler the environment asks for: clang, icx, cl or "".

    ONEDAL_WIN_COMPILER wins, then CC or CXX: one Windows driver compiles both languages.
    """
    override = repo_ctx.os.environ.get("ONEDAL_WIN_COMPILER", "").lower()
    if override:
        return override
    for env_var in ["CC", "CXX"]:
        basename = _driver_basename(repo_ctx.os.environ.get(env_var, ""))
        if basename:
            return basename
    return ""

def _driver_basename(value):
    """The compiler name in a `CC`/`CXX` value, lower case and without `.exe`.

    Unquoted, the program runs up to the first flag, as it may contain spaces (`C:\\Program Files`).
    """
    value = value.strip()
    if value.startswith("\""):
        program = value[1:].split("\"")[0]
    else:
        words = value.split(" ")
        program = words[0]
        for word in words[1:]:
            if word.startswith("-") or word.startswith("/"):
                break
            program = program + " " + word
    basename = program.lower().replace("\\", "/").split("/")[-1]
    if basename.endswith(".exe"):
        basename = basename[:-len(".exe")]
    return basename

def _is_icx_requested(repo_ctx):
    """Return True if icx is requested, or nothing is and icx is on PATH."""
    requested = _requested_compiler(repo_ctx)
    if requested == "icx":
        return True
    if requested:
        return False
    return repo_ctx.which("icx") != None

# The rule each rules_cc toolchain .bzl file provides to BUILD.windows.tpl.
_RULES_CC_TOOLCHAIN_CONFIGS = {
    "windows_cc_toolchain_config.bzl": "cc_toolchain_config",
    "armeabi_cc_toolchain_config.bzl": "armeabi_cc_toolchain_config",
}

def _configure_cc_toolchain_win_cl(repo_ctx):
    """MSVC cl: rules_cc's auto-config, run for the Visual Studio the shell was set up with.

    rules_cc takes the newest Visual Studio unless BAZEL_VC is set; VCINSTALLDIR stands in for it.
    Its files go to `msvc/`, so that `//:all` registers only the root package's toolchain.
    """
    env = dict(repo_ctx.os.environ)
    vc_dir = env.get("VCINSTALLDIR", "").rstrip("\\/")
    if vc_dir and not env.get("BAZEL_VC") and not env.get("BAZEL_VS"):
        env["BAZEL_VC"] = vc_dir

    # The scripts rules_cc runs (get_env.bat, main.cpp) stay in the root, where
    # it executes them; what the BUILD refers to moves with it.
    def file(path, *args, **kwargs):
        prefix = "msvc/" if path.startswith("builtin_include_directory_paths") else ""
        repo_ctx.file(prefix + path, *args, **kwargs)

    def template(path, *args, **kwargs):
        repo_ctx.template("msvc/" + path, *args, **kwargs)

    # These load @bazel_features, which only rules_cc's repository mapping resolves.
    def symlink(target, link_name):
        symbol = _RULES_CC_TOOLCHAIN_CONFIGS.get(link_name)
        if symbol:
            repo_ctx.file("msvc/" + link_name, (
                'load("@rules_cc//cc/private/toolchain:{file}", _impl = "{symbol}")\n' +
                "{symbol} = _impl\n"
            ).format(file = link_name, symbol = symbol))
        else:
            repo_ctx.symlink(target, "msvc/" + link_name)

    configure_windows_toolchain(struct(
        os = struct(environ = env, name = repo_ctx.os.name, arch = repo_ctx.os.arch),
        attr = repo_ctx.attr,
        execute = repo_ctx.execute,
        path = repo_ctx.path,
        which = repo_ctx.which,
        file = file,
        template = template,
        symlink = symlink,
    ))
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:cc_toolchain_win.tpl.BUILD"),
    )

def _configure_cc_toolchain_win_llvm(repo_ctx, reqs, compiler_id):
    """Custom Windows toolchain for icx (`compiler_id = "icx"`) or LLVM clang-cl (`"clang"`)."""
    repo_ctx.file(TEST_CPP_FILE, "int main() { return 0; }")

    tools = (_find_tools_icx(repo_ctx) if compiler_id == "icx"
             else _find_tools_clang(repo_ctx))

    # INCLUDE covers MSVC, the SDK and oneAPI, but not the builtin headers of the compiler package.
    builtin_include_directories = []

    def _add_builtin_include_dir(path):
        p = path.strip().replace("\\", "/")
        if p and p not in builtin_include_directories:
            builtin_include_directories.append(p)
        # setvars.bat can list include dirs that do not exist (tcm 1.6 in Basekit 2026.2),
        # and realpath fails on those.
        if p and repo_ctx.path(p).exists:
            real = str(repo_ctx.path(p).realpath).replace("\\", "/")
            if real and real not in builtin_include_directories:
                builtin_include_directories.append(real)

    for raw in repo_ctx.os.environ.get("INCLUDE", "").split(";"):
        _add_builtin_include_dir(raw)
    cmplr_root = repo_ctx.os.environ.get("CMPLR_ROOT", "").strip().replace("\\", "/")
    if cmplr_root:
        # `latest` may be a junction, and icx reports headers by the resolved versioned path.
        _add_builtin_include_dir(cmplr_root)
        # icx includes these overrides ahead of the system headers.
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

    # clang-cl's builtin headers live in its resource dir, which INCLUDE does not cover.
    if compiler_id == "clang":
        resource_dir = repo_ctx.execute([tools.cc, "-print-resource-dir"])
        if resource_dir.return_code == 0:
            _add_builtin_include_dir("{}/include".format(resource_dir.stdout.strip()))

    reqs_llvm = struct(
        os_id = "win",
        compiler_id = compiler_id,
        dpc_compiler_id = "icpx",
        compiler_version = "local",
        dpc_compiler_version = "local",
        target_arch_id = reqs.target_arch_id,
        host_arch_id = reqs.host_arch_id,
        libc_version = "local",
        libc_abi_version = "local",
        compiler_abi_version = "local",
    )

    dpcc_code_split = "per_kernel"

    compile_flags_cc = get_default_compiler_options(
        repo_ctx, reqs_llvm, tools.cc,
        is_dpcc = False, category = "common",
    )
    compile_flags_dpcc = get_default_compiler_options(
        repo_ctx, reqs_llvm, tools.dpcc,
        is_dpcc = True, category = "common",
    ) if tools.is_dpc_found else []
    if tools.is_dpc_found:
        compile_flags_dpcc = compile_flags_dpcc + add_compiler_option_if_supported(
            repo_ctx, tools.dpcc,
            "-fsycl-device-code-split={}".format(dpcc_code_split),
        )

    compile_flags_pedantic_cc = get_default_compiler_options(
        repo_ctx, reqs_llvm, tools.cc,
        is_dpcc = False, category = "pedantic",
    )
    compile_flags_pedantic_dpcc = get_default_compiler_options(
        repo_ctx, reqs_llvm, tools.dpcc,
        is_dpcc = True, category = "pedantic",
    ) if tools.is_dpc_found else []

    link_flags_cc = []
    # The SYCL import library depends on the CRT, so the toolchain config adds it per build.
    link_flags_dpcc = ([
        "-fsycl",
        "-fsycl-device-code-split={}".format(dpcc_code_split),
    ]) if tools.is_dpc_found else []

    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains:cc_toolchain_win_llvm.tpl.BUILD"),
        {
            "%{cc_toolchain_identifier}": get_toolchain_identifier(reqs_llvm),
            "%{compiler}": compiler_id + "-local",
            "%{abi_version}": "local",
            "%{abi_libc_version}": "local",
            "%{target_libc}": "local",
            "%{target_cpu}": reqs_llvm.target_arch_id,
            "%{host_system_name}": "win-" + reqs_llvm.host_arch_id,
            "%{target_system_name}": "win-" + reqs_llvm.target_arch_id,
            "%{cxx_std_flag_prefix}": "/Qstd:" if compiler_id == "icx" else "/std:",
            "%{openmp_simd_flags}": get_starlark_list(
                ["-Qopenmp-simd"] if compiler_id == "icx" else [],
            ),
            "%{host_cpu_constraint}": ARCH_ID_TO_PLATFORM_CPU[reqs_llvm.host_arch_id],
            "%{target_cpu_constraint}": ARCH_ID_TO_PLATFORM_CPU[reqs_llvm.target_arch_id],
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
                # Debug info in the .obj, as Make's `-DEBC.icx`.
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
                get_cpu_specific_options(reqs_llvm),
            ),
            "%{cpu_flags_dpcc}": get_starlark_list_dict(
                get_cpu_specific_options(reqs_llvm, is_dpcc = True),
            ),
            # Template values land in double-quoted strings, where a backslash starts an escape.
            "%{env_include}": repo_ctx.os.environ.get("INCLUDE", "").replace("\\", "/"),
            "%{env_lib}": repo_ctx.os.environ.get("LIB", "").replace("\\", "/"),
            "%{env_path}": repo_ctx.os.environ.get("PATH", "").replace("\\", "/"),
        },
    )

def _is_clang_requested(repo_ctx):
    """Return True if the Windows toolchain should use upstream LLVM clang-cl."""
    return _requested_compiler(repo_ctx) in ["clang", "clang-cl"]

def configure_cc_toolchain_win(repo_ctx, reqs):
    if reqs.target_arch_id == "arm":
        # Windows ARM64 (Make's `PLAT=winarm`) is clang-cl only: neither icx nor
        # MSVC `cl` compiles the SVE code paths oneDAL selects for this arch.
        requested = _requested_compiler(repo_ctx)
        if requested and requested not in ["clang", "clang-cl"]:
            fail(("Compiler {} (from ONEDAL_WIN_COMPILER or CC/CXX) is not " +
                  "supported when targeting Windows ARM64; only clang-cl is " +
                  "(see dev/make/compiler_definitions/clang.ref.arm.mk)").format(requested))
        _configure_cc_toolchain_win_llvm(repo_ctx, reqs, "clang")
    elif _is_clang_requested(repo_ctx):
        _configure_cc_toolchain_win_llvm(repo_ctx, reqs, "clang")
    elif _is_icx_requested(repo_ctx):
        _configure_cc_toolchain_win_llvm(repo_ctx, reqs, "icx")
    else:
        _configure_cc_toolchain_win_cl(repo_ctx)
