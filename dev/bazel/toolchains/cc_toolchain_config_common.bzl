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

A feature that needs even one different flag belongs in the platform file.
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
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

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

# The actions that run the compiler driver and get their own `action_config`.
_compiler_actions = [
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.cpp_header_parsing,
]

def _flag_set_if(actions, flags, **kwargs):
    """A `flag_set` with one `flag_group`, or with none when `flags` is empty, which it rejects.

    Args:
        actions: the actions the flags apply to.
        flags: the flag list.
        **kwargs: forwarded to `flag_set`, e.g. `with_features`.
    """
    return flag_set(
        actions = actions,
        flag_groups = [flag_group(flags = flags)] if flags else [],
        **kwargs
    )

def _compiler_tools(cc_path, dpcc_path):
    """The host driver for every action, the DPC++ driver for those with the `dpc++` feature.

    Args:
        cc_path: path to the host compiler driver.
        dpcc_path: path to the DPC++ compiler driver; empty when none was found.

    Without a DPC++ driver the host driver stands in, so that a header-only `cc_library` with
    the feature (`@ccl//:headers`) still resolves `$(CC)`; `has_dpc_compiler` keeps it off
    DPC++ sources.
    """
    return [
        tool(
            path = cc_path,
            with_features = [with_feature_set(not_features = ["dpc++"])],
        ),
        tool(
            path = dpcc_path or cc_path,
            with_features = [with_feature_set(features = ["dpc++"])],
        ),
    ]

def _compile_action_configs(compiler_tools, implies):
    """One `action_config` per compile action, all with the same tools and implied features.

    Args:
        compiler_tools: the `tool` list from `compiler_tools`.
        implies: the features every compile action enables.
    """
    return [
        action_config(
            action_name = action_name,
            implies = implies,
            tools = compiler_tools,
        )
        for action_name in _compiler_actions
    ]

def _cpp_link_static_library_action(ar_path):
    """The static library archiving action.

    Args:
        ar_path: path to the archiver, `ar` or `llvm-lib`.
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

def _marker_features(dpcc_path):
    """The flagless features that targets and the build mode request; other flag sets key on them.

    Args:
        dpcc_path: path to the DPC++ compiler driver; empty when none was found.
    """
    return [
        feature(name = "dpc++"),
        feature(name = "has_dpc_compiler", enabled = bool(dpcc_path)),
        feature(name = "c++17"),
        feature(name = "pedantic"),
        feature(name = "dbg"),
        feature(name = "opt"),
    ]

def _no_legacy_features_feature():
    """Opt out of the features Bazel would otherwise add to the toolchain."""
    return feature(name = "no_legacy_features", enabled = True)

def _supports_dynamic_linker_feature():
    """Declare that the toolchain can link shared libraries."""
    return feature(
        name = "supports_dynamic_linker",
        enabled = True,
    )

def _do_not_link_dynamic_dependencies_feature():
    """Declare the opt-in feature that drops dynamic libraries from a link."""
    return feature(
        name = "do_not_link_dynamic_dependencies",
        enabled = False,
    )

def _compiler_input_flags_feature():
    """Pass the source file to the compiler; `-c` works for gcc, icx and clang-cl alike."""
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

def _default_compile_flags_feature(attr):
    """The toolchain's own compile flags, picked by compiler, by mode and by requested feature.

    Args:
        attr: the rule's `ctx.attr`; reads the `compile_flags_*`, `*_compile_flags`,
              `cxx_flags` and `cxx_std_flag_prefix` attributes.
    """
    return feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            _flag_set_if(
                all_compile_actions,
                attr.compile_flags_cc,
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            _flag_set_if(
                all_compile_actions,
                attr.compile_flags_dpcc,
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
            _flag_set_if(
                all_compile_actions,
                attr.dbg_compile_flags,
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            _flag_set_if(
                all_compile_actions,
                attr.opt_compile_flags,
                with_features = [with_feature_set(features = ["opt"])],
            ),
            _flag_set_if(
                all_compile_actions,
                [attr.cxx_std_flag_prefix + "c++17"],
                with_features = [with_feature_set(features = ["c++17"])],
            ),
            _flag_set_if(
                all_compile_actions,
                attr.compile_flags_pedantic_cc,
                with_features = [with_feature_set(features = ["pedantic"],
                                                  not_features = ["dpc++"])],
            ),
            _flag_set_if(
                all_compile_actions,
                attr.compile_flags_pedantic_dpcc,
                with_features = [with_feature_set(features = ["dpc++", "pedantic"])],
            ),
            _flag_set_if(
                all_cpp_compile_actions + [ACTION_NAMES.lto_backend],
                attr.cxx_flags,
            ),
        ],
    )

def _default_link_flags_feature(link_flags_cc, link_flags_dpcc, opt_link_flags):
    """The toolchain's own link flags, picked by compiler and by mode.

    Args:
        link_flags_cc: flags for the host compiler.
        link_flags_dpcc: flags for the DPC++ compiler, used when the `dpc++` feature is on.
        opt_link_flags: flags added on top in the `opt` compilation mode.
    """
    return feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            _flag_set_if(
                all_link_actions + lto_index_actions,
                link_flags_cc,
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            _flag_set_if(
                all_link_actions + lto_index_actions,
                link_flags_dpcc,
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
            _flag_set_if(
                all_link_actions + lto_index_actions,
                opt_link_flags,
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

def _default_dynamic_libraries_feature(dynamic_link_libs):
    """Link the runtime libraries every oneDAL shared library needs.

    Args:
        dynamic_link_libs: the libraries, e.g. `-lstdc++ -lm` on Linux.
    """
    return feature(
        name = "default_dynamic_libraries",
        flag_sets = [
            _flag_set_if(
                all_link_actions + lto_index_actions,
                dynamic_link_libs,
                with_features = [
                    with_feature_set(
                        not_features = ["do_not_link_dynamic_dependencies"],
                    ),
                ],
            ),
        ],
    )

def _cpu_opt_features(cpu_flags_cc, cpu_flags_dpcc):
    """One `<cpu_id>_flags` feature per CPU dispatch variant with its per-ISA compile flags.

    Args:
        cpu_flags_cc: flags for the host compiler by ISA identifier, e.g. `avx2`.
        cpu_flags_dpcc: flags for the DPC++ compiler by the same identifiers.
    """
    return [
        feature(
            name = "{}_flags".format(cpu_id),
            flag_sets = [
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [flag_group(flags = flags_cc)],
                    with_features = [with_feature_set(not_features = ["dpc++"])],
                ),
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [flag_group(flags = cpu_flags_dpcc[cpu_id])],
                    with_features = [with_feature_set(features = ["dpc++"])],
                ),
            ],
        )
        for cpu_id, flags_cc in cpu_flags_cc.items()
    ]

def _create_config_info(ctx, features, action_configs, **kwargs):
    """`cc_common.create_cc_toolchain_config_info` with the identity from the common attributes.

    Args:
        ctx: the rule context.
        features: the `feature` list.
        action_configs: the `action_config` list.
        **kwargs: forwarded, e.g. `artifact_name_patterns`.
    """
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        cxx_builtin_include_directories = ctx.attr.cxx_builtin_include_directories,
        toolchain_identifier = ctx.attr.toolchain_identifier,
        host_system_name = ctx.attr.host_system_name,
        target_system_name = ctx.attr.target_system_name,
        target_cpu = ctx.attr.cpu,
        target_libc = ctx.attr.target_libc,
        compiler = ctx.attr.compiler,
        abi_version = ctx.attr.abi_version,
        abi_libc_version = ctx.attr.abi_libc_version,
        **kwargs
    )

# Attributes declared by both `cc_toolchain_config` rules.
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
    # `-std=` for gcc, clang and icx; clang-cl takes `/Qstd:` (icx) or `/std:`.
    "cxx_std_flag_prefix": attr.string(default = "-std="),
    "link_flags_cc": attr.string_list(),
    "link_flags_dpcc": attr.string_list(),
    "dynamic_link_libs": attr.string_list(),
    "opt_link_flags": attr.string_list(),
    "deterministic_compile_flags": attr.string_list(),
    "cpu_flags_cc": attr.string_list_dict(),
    "cpu_flags_dpcc": attr.string_list_dict(),
}

common = struct(
    compiler_tools = _compiler_tools,
    compile_action_configs = _compile_action_configs,
    cpp_link_static_library_action = _cpp_link_static_library_action,
    marker_features = _marker_features,
    no_legacy_features_feature = _no_legacy_features_feature,
    supports_dynamic_linker_feature = _supports_dynamic_linker_feature,
    do_not_link_dynamic_dependencies_feature =
        _do_not_link_dynamic_dependencies_feature,
    compiler_input_flags_feature = _compiler_input_flags_feature,
    linker_param_file_feature = _linker_param_file_feature,
    user_compile_flags_feature = _user_compile_flags_feature,
    user_link_flags_feature = _user_link_flags_feature,
    default_compile_flags_feature = _default_compile_flags_feature,
    default_link_flags_feature = _default_link_flags_feature,
    default_dynamic_libraries_feature = _default_dynamic_libraries_feature,
    cpu_opt_features = _cpu_opt_features,
    create_config_info = _create_config_info,
)
