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
load("@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "get_starlark_list",
)

TEST_CPP_FILE = "empty.cpp"

_INC_DIR_MARKER_BEGIN = "#include <...>"
_INC_DIR_MARKER_END = "End of search list"

# OSX add " (framework directory)" at the end of line, strip it.
_MAC_FRAMEWORK_SUFFIX = " (framework directory)"
_MAC_FRAMEWORK_SUFFIX_LEN = len(_MAC_FRAMEWORK_SUFFIX)

def detect_os(repo_ctx):
    if "linux" in repo_ctx.os.name:
        return "lnx"
    elif "mac" in repo_ctx.os.name:
        return "mac"
    elif "windows" in repo_ctx.os.name:
        return "win"

def detect_default_compiler(repo_ctx, os_id):
    default = "icx"
    is_icx_available = repo_ctx.which(default) != None
    if not is_icx_available:
        default = {
            "lnx": "gcc",
            "mac": "clang",
            "win": "cl",
        }[os_id]
    return default

def detect_compiler(repo_ctx, os_id):
    if not "CC" in repo_ctx.os.environ:
        return detect_default_compiler(repo_ctx, os_id)
    compiler_path = repo_ctx.os.environ["CC"]
    # TODO: Use more relieble way to detect compiler
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
    elif "icc" in compiler_path:
        return "icc"

def get_starlark_dict(dictionary):
    entries = [ "\"{}\":\"{}\"".format(k, v) for k, v in dictionary.items() ]
    return ",\n    ".join(entries)

def get_starlark_list_dict(dictionary):
    entries = [ "\"{}\": [{}]".format(k, get_starlark_list(v)) for k, v in dictionary.items() ]
    return ",\n    ".join(entries)

def get_cxx_inc_directories(repo_ctx, cc, lang_flag, additional_flags = []):
    """Compute the list of default C++ include directories."""
    result = repo_ctx.execute([cc, "-E", lang_flag, "-", "-v"] + additional_flags)
    index0 = result.stderr.rfind(_INC_DIR_MARKER_BEGIN)
    index1 = result.stderr.find("\n", index0)
    index2 = result.stderr.rfind(_INC_DIR_MARKER_END)
    inc_dirs = result.stderr[index1:index2].strip()
    return [ _prepare_include_path(repo_ctx, p) for p in inc_dirs.split("\n") ]

def get_tmp_dpcpp_inc_directories(repo_ctx, tools):
    return ["/tmp"] if tools.dpc_compiler_version >= "20210803" else []

def is_compiler_option_supported(repo_ctx, cc, option):
    """Checks that `option` is supported by the C compiler."""
    result = repo_ctx.execute([
        cc,
        option,
        "-o",
        "/dev/null",
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
        "/dev/null",
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
    # If the compiler sometimes rewrites paths in the .d files without symlinks
    # (ie when they're shorter), it confuses Bazel's logic for verifying all
    # #included header files are listed as inputs to the action.

    # The '-fno-canonical-system-headers' should be enough, but clang does not
    # support it, so we also try '-no-canonical-prefixes' if first option does
    # not work.
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

def _filter_out_unsupported_compiler_options(repo_ctx, cc, options):
    filtered_options = []
    for option in options:
        filtered_options += add_compiler_option_if_supported(repo_ctx, cc, option)
    return filtered_options


def _prepare_include_path(repo_ctx, path):
    """Resolve and sanitize include path before outputting it into the crosstool.
    Args:
      repo_ctx: repository_ctx object.
      path: an include path to be sanitized.
    Returns:
      Sanitized include path that can be written to the crosstoot. Resulting path
      is absolute if it is outside the repository and relative otherwise.
    """
    path = path.strip()
    if path.endswith(_MAC_FRAMEWORK_SUFFIX):
        path = path[:-_MAC_FRAMEWORK_SUFFIX_LEN].strip()

    # We're on UNIX, so the path delimiter is '/'.
    repo_root = str(repo_ctx.path(".")) + "/"
    path = str(repo_ctx.path(path))
    if path.startswith(repo_root):
        return path[len(repo_root):]
    return path
