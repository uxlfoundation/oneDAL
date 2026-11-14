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
    "@onedal//dev/bazel:flags.bzl",
    "get_default_flags",
    "get_cpu_flags",
)
load("@rules_cc//cc/private/toolchain:lib_cc_configure.bzl",
    "auto_configure_fail",
    "get_starlark_list",
)

TEST_CPP_FILE = "empty.cpp"

_INC_DIR_MARKER_BEGIN = "#include <...>"
_INC_DIR_MARKER_END = "End of search list"

# OSX add " (framework directory)" at the end of line, strip it.
_MAC_FRAMEWORK_SUFFIX = " (framework directory)"
_MAC_FRAMEWORK_SUFFIX_LEN = len(_MAC_FRAMEWORK_SUFFIX)

def arch_dir(os_id, arch_id):
    """The `lib/<arch>` directory of the release tree, Make's ARCH_DIR_ONEDAL."""
    return "ARM64" if arch_id == "arm" and os_id == "win" else arch_id

def detect_os(repo_ctx):
    if "linux" in repo_ctx.os.name:
        return "lnx"
    elif "mac" in repo_ctx.os.name:
        return "mac"
    elif "windows" in repo_ctx.os.name:
        return "win"

def detect_default_compiler(repo_ctx, os_id):
    candidates = {
        "lnx": ["icx", "gcc", "clang"],
        "mac": ["icx", "clang"],
        "win": ["icx", "cl"],
    }[os_id]
    for compiler in candidates:
        if repo_ctx.which(compiler) != None:
            return compiler

    # None found: return the usual default so the toolchain reports it missing.
    return candidates[1]

def detect_compiler(repo_ctx, os_id):
    if not "CC" in repo_ctx.os.environ:
        return detect_default_compiler(repo_ctx, os_id)
    compiler_path = repo_ctx.os.environ["CC"]
    # TODO: detect the compiler by running it rather than by its name.
    if "gcc" in compiler_path:
        return "gcc"
    elif "clang" in compiler_path:
        return "clang"
    elif "cl" in compiler_path:
        return "cl"
    elif "icx" in compiler_path:
        return "icx"
    elif "icpx" in compiler_path:
        return "icpx"
    if os_id == "lnx":
        fail("CC={} is not a supported compiler; set CC to gcc, clang or icx ".format(compiler_path) +
             "(a versioned name such as gcc-14 or clang-19, or a full path, also works), " +
             "or unset it to pick the first of icx, gcc and clang found in PATH.")
    return None

def is_dpc_compiler_found(repo_ctx, os_id):
    """Returns whether the DPC++ targets can be built: icpx in PATH on Linux, always on macOS."""
    if os_id != "lnx":
        return True
    return repo_ctx.which("icpx") != None

def find_tool(repo_ctx, tool_name, mandatory = False):
    """Returns `(path, found)` of a tool given as an absolute path or looked up in PATH.

    Args:
        repo_ctx: the repository context.
        tool_name: the absolute path or the name to look up in PATH.
        mandatory: fail when the tool is missing instead of returning `("", False)`.

    The empty path of a missing optional tool registers no tool for its actions,
    so a target that needs it fails during analysis.
    """
    if tool_name.startswith("/"):
        tool_path = repo_ctx.path(tool_name) if repo_ctx.path(tool_name).exists else None
    else:
        tool_path = repo_ctx.which(tool_name)
    if tool_path == None:
        if mandatory:
            auto_configure_fail("Cannot find {}; try to correct your $PATH".format(tool_name))
        return "", False
    return str(tool_path), True

# oneDAL arch IDs (Make: ARCH_DIR_ONEDAL) by the machine field of a cross-compiler triple.
_ARCH_ID_BY_TRIPLE_PREFIX = {
    "aarch64": "arm",
    # Not GNU, but clang accepts `arm64-pc-windows-msvc`.
    "arm64": "arm",
    "riscv64": "riscv64",
    "x86_64": "intel64",
    "amd64": "intel64",
}

# By `uname -m`, which some ARM hosts report as `arm64`.
_ARCH_ID_BY_UNAME = dict(_ARCH_ID_BY_TRIPLE_PREFIX)
_ARCH_ID_BY_UNAME["arm64"] = "arm"

def _arch_id_from_uname(value):
    return _ARCH_ID_BY_UNAME.get(value)

# By `PROCESSOR_ARCHITECTURE`, as the Windows CI scripts detect the host.
_ARCH_ID_BY_PROCESSOR_ARCHITECTURE = {
    "AMD64": "intel64",
    "ARM64": "arm",
}

def _triple_prefix_to_triple(basename):
    """`aarch64-linux-gnu-gcc-13` -> `aarch64-linux-gnu-`; the version suffix is optional."""
    parts = basename.split("-")
    if parts[-1].isdigit() and len(parts) > 2:
        parts = parts[:-2]
    else:
        parts = parts[:-1]
    return "-".join(parts) + "-"

def _detect_cross_compiler(repo_ctx):
    """Returns `(triple_prefix, arch_id)` from a cross-compiler `CC`/`CXX`, or `(None, None)`."""
    for env_var in ["CC", "CXX"]:
        basename = repo_ctx.os.environ.get(env_var, "").split("/")[-1]
        for prefix, arch_id in _ARCH_ID_BY_TRIPLE_PREFIX.items():
            if basename.startswith(prefix + "-"):
                return _triple_prefix_to_triple(basename), arch_id
    return None, None

def _detect_target_flag_arch(repo_ctx):
    """Returns the arch ID of a `--target=<triple>` in `CC`/`CXX`, or `None`.

    Unlike a triple prefix it says nothing about the tool names, so get_cross_tool_prefix ignores it.
    """
    for env_var in ["CC", "CXX"]:
        for arg in repo_ctx.os.environ.get(env_var, "").split(" "):
            if arg.startswith("--target="):
                machine = arg[len("--target="):].split("-")[0]
                arch_id = _ARCH_ID_BY_TRIPLE_PREFIX.get(machine)
                if arch_id:
                    return arch_id
    return None

def get_cross_tool_prefix(repo_ctx):
    """Returns the triple prefix (`aarch64-linux-gnu-`) of a cross-compiler `CC`/`CXX`, or `""`."""
    triple_prefix, _ = _detect_cross_compiler(repo_ctx)
    return triple_prefix or ""

ARCH_ID_TO_PLATFORM_CPU = {
    "intel64": "x86_64",
    "arm": "aarch64",
    "riscv64": "riscv64",
}

def detect_host_arch(repo_ctx, os_id):
    """Detects the oneDAL arch ID (intel64/arm/riscv64) of the exec host."""
    if os_id == "win":
        # These describe the current process: an emulated x86_64 process on ARM64 reports AMD64.
        # Use a `--target=` triple in CC/CXX there; repository rules never see `--platforms`.
        for env_var in ["PROCESSOR_ARCHITEW6432", "PROCESSOR_ARCHITECTURE"]:
            value = repo_ctx.os.environ.get(env_var, "").strip().upper()
            arch_id = _ARCH_ID_BY_PROCESSOR_ARCHITECTURE.get(value)
            if arch_id:
                return arch_id
        return "intel64"
    if os_id != "lnx":
        # oneDAL supports macOS on x86 only.
        return "intel64"
    result = repo_ctx.execute(["uname", "-m"])
    arch_id = _arch_id_from_uname(result.stdout.strip()) if result.return_code == 0 else None
    return arch_id or "intel64"

def detect_target_arch(repo_ctx, host_arch_id):
    """Detects the target arch ID from a cross-compiler or `--target=` in CC/CXX, else the host's."""
    _, arch_id = _detect_cross_compiler(repo_ctx)
    return arch_id or _detect_target_flag_arch(repo_ctx) or host_arch_id

def get_starlark_list_dict(dictionary):
    entries = [ "\"{}\": [{}]".format(k, get_starlark_list(v)) for k, v in dictionary.items() ]
    return ",\n    ".join(entries)

def get_cxx_inc_directories(repo_ctx, cc, lang_flag, additional_flags = []):
    """Compute the list of default C++ include directories."""
    result = repo_ctx.execute([cc, "-E", lang_flag, "-", "-v"] + additional_flags)
    index0 = result.stderr.rfind(_INC_DIR_MARKER_BEGIN)
    index2 = result.stderr.rfind(_INC_DIR_MARKER_END)
    if index0 == -1 or index2 == -1:
        # A failed probe would add the repository's absolute path, tying every action key to
        # the output base.
        return []
    index1 = result.stderr.find("\n", index0)
    inc_dirs = result.stderr[index1:index2].strip()
    return [ _prepare_include_path(repo_ctx, p) for p in inc_dirs.split("\n") if p.strip() ]

def _null_device(repo_ctx):
    return "NUL" if "windows" in repo_ctx.os.name else "/dev/null"

def is_compiler_option_supported(repo_ctx, cc, option):
    """Checks that `option` is supported by the C compiler."""
    result = repo_ctx.execute([
        cc,
        option,
        "-o",
        _null_device(repo_ctx),
        "-c",
        str(repo_ctx.path(TEST_CPP_FILE)),
    ])
    return (result.stderr.find(option) == -1 and
            result.stderr.find("warning:") == -1)


def is_linker_option_supported(repo_ctx, cc, option, pattern):
    """Checks that `option` is supported by the C linker."""
    result = repo_ctx.execute([
        cc,
        option,
        "-o",
        _null_device(repo_ctx),
        str(repo_ctx.path(TEST_CPP_FILE)),
    ])
    return result.stderr.find(pattern) == -1


def add_compiler_option_if_supported(repo_ctx, cc, option):
    """Returns `[option]` if supported, `[]` otherwise."""
    return [option] if is_compiler_option_supported(repo_ctx, cc, option) else []


def add_linker_option_if_supported(repo_ctx, cc, option, pattern):
    """Returns `[option]` if supported, `[]` otherwise."""
    return [option] if is_linker_option_supported(repo_ctx, cc, option, pattern) else []


def get_no_canonical_prefixes_opt(repo_ctx, cc):
    # Paths a compiler rewrites in the .d files break Bazel's check that every header is an input.
    # clang lacks -fno-canonical-system-headers, so fall back to -no-canonical-prefixes.
    opt = add_compiler_option_if_supported(
        repo_ctx,
        cc,
        "-fno-canonical-system-headers",
    )
    if len(opt) == 0:
        return add_compiler_option_if_supported(
            repo_ctx,
            cc,
            "-no-canonical-prefixes",
        )
    return opt


def get_toolchain_identifier(reqs):
    return "{}-{}-{}-{}".format(reqs.os_id, reqs.target_arch_id,
                                reqs.compiler_id, reqs.compiler_version)


def get_default_compiler_options(repo_ctx, reqs, cc, is_dpcc=False, category="common"):
    options = _get_unfiltered_default_compiler_options(reqs, is_dpcc, category)
    return _filter_out_unsupported_compiler_options(repo_ctx, cc, options)

def get_cpu_specific_options(reqs, is_dpcc=False):
    compiler_id = reqs.dpc_compiler_id if is_dpcc else reqs.compiler_id
    return get_cpu_flags(reqs.target_arch_id, reqs.os_id, compiler_id)

def _get_unfiltered_default_compiler_options(reqs, is_dpcc, category):
    compiler_id = reqs.dpc_compiler_id if is_dpcc else reqs.compiler_id
    return get_default_flags(reqs.target_arch_id, reqs.os_id, compiler_id, category)

# The probe tests options against the default target, where a cross target's `-march` fails;
# Make passes these unconditionally.
_UNFILTERED_OPTION_PREFIXES = [
    "--target=",
    "-march=",
    "-mcpu=",
]

def _is_unfiltered_option(option):
    for prefix in _UNFILTERED_OPTION_PREFIXES:
        if option.startswith(prefix):
            return True
    return False

def _filter_out_unsupported_compiler_options(repo_ctx, cc, options):
    filtered_options = []
    for option in options:
        if _is_unfiltered_option(option):
            filtered_options.append(option)
        else:
            filtered_options += add_compiler_option_if_supported(repo_ctx, cc, option)
    return filtered_options


def _prepare_include_path(repo_ctx, path):
    """Returns the absolute include path, without the macOS framework suffix."""
    path = path.strip()
    if path.endswith(_MAC_FRAMEWORK_SUFFIX):
        path = path[:-_MAC_FRAMEWORK_SUFFIX_LEN].strip()
    return str(repo_ctx.path(path))
