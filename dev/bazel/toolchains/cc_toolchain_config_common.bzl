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

"""Parts shared by the Linux and Windows `cc_toolchain_config` implementations.

`cc_toolchain_config_lnx.bzl` and `cc_toolchain_config_win.bzl` describe two
different compiler drivers, so most of their features differ in the flags they
emit (`-I` against `/I`, `-MD -MF` against `/clang:-MD /clang:-MF`, and so on)
and have to stay apart. What does not differ is collected here:

* the action groups the flag sets are attached to, which are simply lists of
  `ACTION_NAMES` and were identical in both files;
* the features whose flags come entirely from rule attributes or from Bazel
  build variables, and therefore contain nothing platform specific;
* the rule attributes both `cc_toolchain_config` rules declare.

Anything added here must hold for both platforms. A feature that needs even one
different flag belongs in the platform file instead -- a shared factory with a
per-platform branch inside would be harder to follow than the duplicate it
replaces.
"""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "action_config",
    "feature",
    "flag_group",
    "flag_set",
    "tool",
    "with_feature_set",
)

all_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
    ACTION_NAMES.lto_backend,
]

all_cpp_compile_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.clif_match,
]

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

lto_index_actions = [
    ACTION_NAMES.lto_index_for_executable,
    ACTION_NAMES.lto_index_for_dynamic_library,
    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
]

def _cpp_link_static_library_action(ar_path):
    """Describe the static library archiving action.

    Args:
        ar_path: path to the archiver, `ar` or `xilib`.

    Returns:
        The `action_config`.
    """
    return action_config(
        action_name = ACTION_NAMES.cpp_link_static_library,
        implies = [
            "archiver_flags",
            "linker_param_file",
        ],
        tools = [
            tool(path = ar_path),
        ],
    )

def _supports_dynamic_linker_feature():
    """Declare that the toolchain can link shared libraries."""
    return feature(
        name = "supports_dynamic_linker",
        enabled = True,
    )

def _do_not_link_dynamic_dependencies_feature():
    """Declare the opt-in feature that drops dynamic libraries from a link.

    Nothing enables it by default; see the comment in `_cc_dynamic_lib_impl`
    (`dev/bazel/cc.bzl`) for why the released libraries must keep their
    DT_NEEDED entries. It stays defined so that it remains reachable through
    `--features`.
    """
    return feature(
        name = "do_not_link_dynamic_dependencies",
        enabled = False,
    )

def _compiler_input_flags_feature():
    """Pass the source file to the compiler.

    `-c` is spelled the same way by gcc and by icx on Windows, which accepts
    both the GNU and the MSVC spelling, so this needs no platform variant.
    """
    return feature(
        name = "compiler_input_flags",
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-c", "%{source_file}"],
                        expand_if_available = "source_file",
                    ),
                ],
            ),
        ],
    )

def _linker_param_file_feature():
    """Let the linker and the archiver read their inputs from a param file."""
    return feature(
        name = "linker_param_file",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["@%{linker_param_file}"],
                        expand_if_available = "linker_param_file",
                    ),
                ],
            ),
        ],
    )

def _user_compile_flags_feature():
    """Append the flags a target asks for through `copts`."""
    return feature(
        name = "user_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

def _user_link_flags_feature():
    """Append the flags a target asks for through `linkopts`."""
    return feature(
        name = "user_link_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_link_flags}"],
                        iterate_over = "user_link_flags",
                        expand_if_available = "user_link_flags",
                    ),
                ],
            ),
        ],
    )

def _default_link_flags_feature(link_flags_cc, link_flags_dpcc, opt_link_flags):
    """Apply the toolchain's own link flags, picked by compiler and by mode.

    Args:
        link_flags_cc: flags for the host compiler.
        link_flags_dpcc: flags for the DPC++ compiler, used when the `dpc++`
                         feature is on.
        opt_link_flags: flags added on top in the `opt` compilation mode.

    Returns:
        The `feature`.
    """
    return feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([
                    flag_group(
                        flags = link_flags_cc,
                    ),
                ] if link_flags_cc else []),
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([
                    flag_group(
                        flags = link_flags_dpcc,
                    ),
                ] if link_flags_dpcc else []),
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([
                    flag_group(
                        flags = opt_link_flags,
                    ),
                ] if opt_link_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

def _default_dynamic_libraries_feature(dynamic_link_libs):
    """Link the runtime libraries every oneDAL shared library needs.

    Args:
        dynamic_link_libs: the libraries, e.g. `-lstdc++ -lm` on Linux.

    Returns:
        The `feature`.
    """
    return feature(
        name = "default_dynamic_libraries",
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = (
                    [flag_group(flags = dynamic_link_libs)]
                    if dynamic_link_libs else []
                ),
                with_features = [
                    with_feature_set(
                        not_features = ["do_not_link_dynamic_dependencies"],
                    ),
                ],
            ),
        ],
    )

def _cpu_opt_feature(cpu_id, cpu_flags_cc, cpu_flags_dpcc):
    """Describe the per-ISA compile flags of a single CPU dispatch variant.

    Args:
        cpu_id: the ISA identifier, e.g. `avx2`; names the feature.
        cpu_flags_cc: flags for the host compiler.
        cpu_flags_dpcc: flags for the DPC++ compiler.

    Returns:
        The `feature`, named `<cpu_id>_flags`.
    """
    return feature(
        name = "{}_flags".format(cpu_id),
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = cpu_flags_cc)],
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = cpu_flags_dpcc)],
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
        ],
    )

# Attributes declared by both `cc_toolchain_config` rules. Each platform adds
# its own on top, see the `rule()` calls in the two files.
COMMON_ATTRS = {
    "cpu": attr.string(mandatory = True),
    "compiler": attr.string(mandatory = True),
    "toolchain_identifier": attr.string(mandatory = True),
    "host_system_name": attr.string(mandatory = True),
    "target_system_name": attr.string(mandatory = True),
    "target_libc": attr.string(mandatory = True),
    "abi_version": attr.string(mandatory = True),
    "abi_libc_version": attr.string(mandatory = True),
    "cc_path": attr.string(mandatory = True),
    "dpcc_path": attr.string(mandatory = True),
    "ar_path": attr.string(mandatory = True),
    "cxx_builtin_include_directories": attr.string_list(),
    "compile_flags_cc": attr.string_list(),
    "compile_flags_dpcc": attr.string_list(),
    "compile_flags_pedantic_cc": attr.string_list(),
    "compile_flags_pedantic_dpcc": attr.string_list(),
    "dbg_compile_flags": attr.string_list(),
    "opt_compile_flags": attr.string_list(),
    "cxx_flags": attr.string_list(),
    "link_flags_cc": attr.string_list(),
    "link_flags_dpcc": attr.string_list(),
    "dynamic_link_libs": attr.string_list(),
    "opt_link_flags": attr.string_list(),
    "deterministic_compile_flags": attr.string_list(),
    "cpu_flags_cc": attr.string_list_dict(),
    "cpu_flags_dpcc": attr.string_list_dict(),
}

common = struct(
    cpp_link_static_library_action = _cpp_link_static_library_action,
    supports_dynamic_linker_feature = _supports_dynamic_linker_feature,
    do_not_link_dynamic_dependencies_feature =
        _do_not_link_dynamic_dependencies_feature,
    compiler_input_flags_feature = _compiler_input_flags_feature,
    linker_param_file_feature = _linker_param_file_feature,
    user_compile_flags_feature = _user_compile_flags_feature,
    user_link_flags_feature = _user_link_flags_feature,
    default_link_flags_feature = _default_link_flags_feature,
    default_dynamic_libraries_feature = _default_dynamic_libraries_feature,
    cpu_opt_feature = _cpu_opt_feature,
)
