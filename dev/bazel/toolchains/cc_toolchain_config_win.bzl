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

# Toolchain config for clang-cl on Windows (icx and LLVM clang-cl), in clang-cl flag spellings.
# Arch-independent: the linker and archiver take the machine type from the objects.

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")

load("@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "env_entry",
    "env_set",
    "variable_with_value",
    "with_feature_set",
    "action_config",
    "artifact_name_pattern",
)
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")
load("@onedal//dev/bazel/config:config.bzl", "ConfigFlagInfo")
load("@onedal//dev/bazel/toolchains:cc_toolchain_config_common.bzl",
    "common",
    "all_compile_actions",
    "all_link_actions",
    "lto_index_actions",
    "COMMON_ATTRS",
)

def _impl(ctx):
    compiler_tools = common.compiler_tools(ctx.attr.cc_path, ctx.attr.dpcc_path)
    linker_tools = common.compiler_tools(ctx.attr.cc_link_path, ctx.attr.dpcc_link_path)

    # --- action configs -----------------------------------------------------

    compile_actions = common.compile_action_configs(compiler_tools, [
        "default_compile_flags",
        "user_compile_flags",
        "unfiltered_compile_flags",
        "compiler_input_flags",
        "compiler_output_flags",
    ])

    common_link_implies = [
        "default_link_flags",
        "user_link_flags",
        "output_execpath_flags",
        "libraries_to_link",
        "library_search_directories",
        "linker_param_file",
    ]

    cpp_link_executable_action = action_config(
        action_name = ACTION_NAMES.cpp_link_executable,
        implies = common_link_implies + ["default_dynamic_libraries"],
        tools = linker_tools,
    )
    cpp_link_nodeps_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
        implies = common_link_implies + ["shared_flag"],
        tools = linker_tools,
    )
    cpp_link_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_dynamic_library,
        implies = common_link_implies + ["shared_flag", "default_dynamic_libraries"],
        tools = linker_tools,
    )
    cpp_link_static_library_action = common.cpp_link_static_library_action(ctx.attr.ar_path)

    action_configs = compile_actions + [
        cpp_link_executable_action,
        cpp_link_nodeps_dynamic_library_action,
        cpp_link_dynamic_library_action,
        cpp_link_static_library_action,
    ]

    # --- features -----------------------------------------------------------

    # Debug CRT (Make MSVC_RUNTIME_VERSION=debug), separate from `dbg` as it renames the libraries.
    # It follows @config//:msvc_runtime, as do the library names and the dependency selects.
    msvc_runtime_debug_feature = feature(
        name = "msvc_runtime_debug",
        enabled = ctx.attr._msvc_runtime[ConfigFlagInfo].flag == "debug",
    )

    # Without Make's `-Zl` the objects keep their /defaultlib directives, so -MD/-MDd alone picks
    # the CRT and the CRT import libraries need not be listed.
    runtime_library_feature = feature(
        name = "runtime_library",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["-MDd"])],
                with_features = [with_feature_set(features = ["msvc_runtime_debug"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["-MD"])],
                with_features = [with_feature_set(not_features = ["msvc_runtime_debug"])],
            ),
            # icx rejects OpenMP SIMD with the debug CRT (-Wdebug-option-simd); Make: COMPILER.win.icx.
            # Empty for clang-cl, and flag_group refuses an empty `flags` list.
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([flag_group(flags = ctx.attr.openmp_simd_flags)]
                               if ctx.attr.openmp_simd_flags else []),
                with_features = [with_feature_set(
                    not_features = ["msvc_runtime_debug"],
                )],
            ),
            # Make: `-MDd /debug:none` in dpcpp.mk.
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["/debug:none"])],
                with_features = [with_feature_set(
                    features = ["dpc++", "msvc_runtime_debug"],
                )],
            ),
        ],
    )

    # icx omits the SYCL import library when the device objects come in a /WHOLEARCHIVE
    # library, leaving __sycl_register_lib unresolved. Make: sycl$d.lib.
    sycl_runtime_library_feature = feature(
        name = "sycl_runtime_library",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [flag_group(flags = ["sycld.lib"])],
                with_features = [with_feature_set(
                    features = ["dpc++", "msvc_runtime_debug"],
                )],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [flag_group(flags = ["sycl.lib"])],
                with_features = [with_feature_set(
                    features = ["dpc++"],
                    not_features = ["msvc_runtime_debug"],
                )],
            ),
        ],
    )
    # No `supports_pic`: PE/COFF has no PIC variant, and cc_module rejects two object variants.
    supports_dynamic_linker_feature = common.supports_dynamic_linker_feature()
    do_not_link_dynamic_dependencies_feature = common.do_not_link_dynamic_dependencies_feature()
    no_legacy_features_feature = common.no_legacy_features_feature()

    # With these two features cc_common.link declares the DLL's import library, as for rules_cc's MSVC.
    # has_configured_linker_path keeps the action config's linker instead of Bazel's wrapper script.
    targets_windows_feature = feature(name = "targets_windows", enabled = True)
    supports_interface_shared_libraries_feature = feature(
        name = "supports_interface_shared_libraries",
        enabled = True,
    )
    has_configured_linker_path_feature = feature(
        name = "has_configured_linker_path",
        enabled = True,
    )
    interface_library_output_flags_feature = feature(
        name = "interface_library_output_flags",
        enabled = True,
        flag_sets = [flag_set(
            actions = all_link_actions + lto_index_actions,
            flag_groups = [flag_group(
                flags = ["/IMPLIB:%{interface_library_output_path}"],
                expand_if_available = "interface_library_output_path",
            )],
        )],
    )

    # INCLUDE and LIB from the configure-time shell, so actions find the MSVC, SDK and oneAPI files.
    msvc_env_entries = []
    if ctx.attr.env_include:
        msvc_env_entries.append(env_entry(key = "INCLUDE", value = ctx.attr.env_include))
    if ctx.attr.env_lib:
        msvc_env_entries.append(env_entry(key = "LIB", value = ctx.attr.env_lib))
    if ctx.attr.env_path:
        msvc_env_entries.append(env_entry(key = "PATH", value = ctx.attr.env_path))
    msvc_env_feature = feature(
        name = "msvc_env",
        enabled = True,
        env_sets = [env_set(
            actions = all_compile_actions + all_link_actions + [
                ACTION_NAMES.cpp_link_static_library,
            ] + lto_index_actions,
            env_entries = msvc_env_entries,
        )] if msvc_env_entries else [],
    )

    compiler_input_flags_feature = common.compiler_input_flags_feature()

    compiler_output_flags_feature = feature(
        name = "compiler_output_flags",
        flag_sets = [flag_set(
            actions = all_compile_actions,
            flag_groups = [
                flag_group(
                    flags = ["/Fa%{output_file}"],
                    expand_if_available = "output_assembly_file",
                ),
                flag_group(
                    flags = ["/P", "/Fi%{output_file}"],
                    expand_if_available = "output_preprocess_file",
                ),
                flag_group(
                    flags = ["/Fo%{output_file}"],
                    expand_if_available = "output_file",
                ),
            ],
        )],
    )

    linker_param_file_feature = common.linker_param_file_feature()

    default_compile_flags_feature = common.default_compile_flags_feature(ctx.attr)

    default_link_flags_feature = common.default_link_flags_feature(
        ctx.attr.link_flags_cc,
        ctx.attr.link_flags_dpcc,
        ctx.attr.opt_link_flags,
    )

    user_compile_flags_feature = common.user_compile_flags_feature()

    user_link_flags_feature = common.user_link_flags_feature()

    default_dynamic_libraries_feature = common.default_dynamic_libraries_feature(
        ctx.attr.dynamic_link_libs,
    )

    # DPC++ links go through icx so it adds the SYCL device libraries.
    dpc_linker_mode_feature = feature(
        name = "dpc_linker_mode",
        enabled = True,
        flag_sets = [flag_set(
            actions = all_link_actions + lto_index_actions,
            flag_groups = [flag_group(flags = ["/link"])],
            with_features = [with_feature_set(features = ["dpc++"])],
        )],
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        enabled = True,
        flag_sets = [flag_set(
            actions = all_compile_actions,
            flag_groups = ([flag_group(flags = ctx.attr.deterministic_compile_flags)]
                           if ctx.attr.deterministic_compile_flags else []),
        )],
    )

    # clang-cl forwards `-libpath:` to lld-link; a `/` prefix would be taken for an input file.
    library_search_directories_feature = feature(
        name = "library_search_directories",
        flag_sets = [flag_set(
            actions = all_link_actions + lto_index_actions,
            flag_groups = [flag_group(
                flags = ["-libpath:%{library_search_directories}"],
                iterate_over = "library_search_directories",
                expand_if_available = "library_search_directories",
            )],
        )],
    )

    preprocessor_defines_feature = feature(
        name = "preprocessor_defines",
        enabled = True,
        flag_sets = [flag_set(
            actions = [
                ACTION_NAMES.preprocess_assemble,
                ACTION_NAMES.linkstamp_compile,
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
                ACTION_NAMES.clif_match,
            ],
            flag_groups = [flag_group(
                flags = ["/D%{preprocessor_defines}"],
                iterate_over = "preprocessor_defines",
            )],
        )],
    )

    includes_feature = feature(
        name = "includes",
        enabled = True,
        flag_sets = [flag_set(
            actions = [
                ACTION_NAMES.preprocess_assemble,
                ACTION_NAMES.linkstamp_compile,
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
                ACTION_NAMES.clif_match,
            ],
            flag_groups = [flag_group(
                flags = ["/FI", "%{includes}"],
                iterate_over = "includes",
                expand_if_available = "includes",
            )],
        )],
    )

    # clang-cl has no quote-include flag, so quote includes become `/I`.
    include_paths_feature = feature(
        name = "include_paths",
        enabled = True,
        flag_sets = [flag_set(
            actions = [
                ACTION_NAMES.preprocess_assemble,
                ACTION_NAMES.linkstamp_compile,
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.cpp_module_compile,
                ACTION_NAMES.clif_match,
            ],
            flag_groups = [
                flag_group(
                    flags = ["/I%{quote_include_paths}"],
                    iterate_over = "quote_include_paths",
                ),
                flag_group(
                    flags = ["/I%{include_paths}"],
                    iterate_over = "include_paths",
                ),
                flag_group(
                    flags = ["/imsvc", "%{system_include_paths}"],
                    iterate_over = "system_include_paths",
                ),
            ],
        )],
    )

    libraries_to_link_feature = feature(
        name = "libraries_to_link",
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [flag_group(
                    iterate_over = "libraries_to_link",
                    flag_groups = [
                        flag_group(
                            flags = ["%{libraries_to_link.object_files}"],
                            iterate_over = "libraries_to_link.object_files",
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file_group",
                            ),
                        ),
                        flag_group(
                            flags = ["%{libraries_to_link.name}"],
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file",
                            ),
                        ),
                        flag_group(
                            flags = ["%{libraries_to_link.name}"],
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "static_library",
                            ),
                        ),
                    ],
                    expand_if_available = "libraries_to_link",
                )],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [flag_group(
                    iterate_over = "libraries_to_link",
                    flag_groups = [
                        flag_group(
                            flags = ["%{libraries_to_link.name}"],
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "interface_library",
                            ),
                        ),
                        # Dynamic libraries link through their import .lib.
                        flag_group(
                            flags = ["%{libraries_to_link.name}.lib"],
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "dynamic_library",
                            ),
                        ),
                    ],
                    expand_if_available = "libraries_to_link",
                )],
                with_features = [with_feature_set(
                    not_features = ["do_not_link_dynamic_dependencies"],
                )],
            ),
        ],
    )

    archiver_flags_feature = feature(
        name = "archiver_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [flag_group(
                    flags = ["/OUT:%{output_execpath}"],
                    expand_if_available = "output_execpath",
                )],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [flag_group(
                    iterate_over = "libraries_to_link",
                    flag_groups = [
                        flag_group(
                            flags = ["%{libraries_to_link.name}"],
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file",
                            ),
                        ),
                        flag_group(
                            flags = ["%{libraries_to_link.object_files}"],
                            iterate_over = "libraries_to_link.object_files",
                            expand_if_equal = variable_with_value(
                                name = "libraries_to_link.type",
                                value = "object_file_group",
                            ),
                        ),
                    ],
                    expand_if_available = "libraries_to_link",
                )],
            ),
        ],
    )

    # Passed through `/clang:`, because `-MD` in clang-cl selects the CRT.
    dependency_file_feature = feature(
        name = "dependency_file",
        enabled = True,
        flag_sets = [flag_set(
            actions = [
                ACTION_NAMES.assemble,
                ACTION_NAMES.preprocess_assemble,
                ACTION_NAMES.c_compile,
                ACTION_NAMES.cpp_compile,
                ACTION_NAMES.cpp_module_compile,
                ACTION_NAMES.cpp_header_parsing,
                ACTION_NAMES.clif_match,
            ],
            flag_groups = [flag_group(
                flags = [
                    "/clang:-MD",
                    "/clang:-MF",
                    "/clang:%{dependency_file}",
                ],
                expand_if_available = "dependency_file",
            )],
        )],
    )

    output_execpath_flags_feature = feature(
        name = "output_execpath_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [flag_group(
                    flags = ["/OUT:%{output_execpath}"],
                    expand_if_available = "output_execpath",
                )],
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [flag_group(
                    flags = ["-o%{output_execpath}"],
                    expand_if_available = "output_execpath",
                )],
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
        ],
    )

    # Non-DPC links run lld-link and take `/DLL`; DPC++ links pass `-LD` to the icx driver, as Make.
    shared_flag_feature = feature(
        name = "shared_flag",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["/DLL"])],
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.lto_index_for_dynamic_library,
                    ACTION_NAMES.lto_index_for_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["-LD"])],
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
        ],
    )

    # --- assemble feature list ---------------------------------------------

    features = [
        no_legacy_features_feature,
        msvc_env_feature,
    ]
    features.extend(common.marker_features(ctx.attr.dpcc_path))
    features.extend([
        msvc_runtime_debug_feature,
        runtime_library_feature,
        supports_dynamic_linker_feature,
        targets_windows_feature,
        supports_interface_shared_libraries_feature,
        has_configured_linker_path_feature,
        do_not_link_dynamic_dependencies_feature,
        default_compile_flags_feature,
    ])
    features.extend(common.cpu_opt_features(ctx.attr.cpu_flags_cc, ctx.attr.cpu_flags_dpcc))
    features.extend([
        user_compile_flags_feature,
        preprocessor_defines_feature,
        includes_feature,
        include_paths_feature,
        unfiltered_compile_flags_feature,
        dependency_file_feature,
        compiler_input_flags_feature,
        compiler_output_flags_feature,
        default_link_flags_feature,
        # Before `/link`: on a DPC++ link this is the only input the icx driver sees,
        # which otherwise fails with "no input files".
        sycl_runtime_library_feature,
        # The icx driver flags go before the `/link` from dpc_linker_mode and the
        # lld-link inputs after it; clang-cl ignores lld-link options.
        shared_flag_feature,
        output_execpath_flags_feature,
        dpc_linker_mode_feature,
        # `/IMPLIB:` is a linker option, so it sits after `/link` on a DPC++
        # link and overrides the `-implib:` the icx driver derives from `-o`.
        interface_library_output_flags_feature,
        library_search_directories_feature,
        libraries_to_link_feature,
        user_link_flags_feature,
        default_dynamic_libraries_feature,
        archiver_flags_feature,
        linker_param_file_feature,
    ])

    artifact_name_patterns = [
        artifact_name_pattern(
            category_name = "object_file",
            prefix = "",
            extension = ".obj",
        ),
        artifact_name_pattern(
            category_name = "static_library",
            prefix = "",
            extension = ".lib",
        ),
        artifact_name_pattern(
            category_name = "executable",
            prefix = "",
            extension = ".exe",
        ),
        artifact_name_pattern(
            category_name = "dynamic_library",
            prefix = "",
            extension = ".dll",
        ),
        artifact_name_pattern(
            category_name = "interface_library",
            prefix = "",
            # Bazel accepts only `.ifso`, `.tbd`, `.if.lib` or `.lib`; the release rule renames to `_dll.lib`.
            extension = ".if.lib",
        ),
    ]

    return common.create_config_info(
        ctx,
        features,
        action_configs,
        artifact_name_patterns = artifact_name_patterns,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = dict(
        COMMON_ATTRS,
        # A standalone linker (`lld-link` or `link`); Linux links through the compiler driver.
        cc_link_path = attr.string(mandatory = True),
        dpcc_link_path = attr.string(mandatory = True),
        # `/Qstd:` is Intel-only; clang-cl warns on it, which -Werror makes fatal.
        cxx_std_flag_prefix = attr.string(default = "/Qstd:"),
        # `-Qopenmp-simd` for icx, empty for clang-cl; release CRT only.
        openmp_simd_flags = attr.string_list(),
        # Captured at configure time, so actions find the MSVC, SDK and oneAPI files without the shell env.
        env_include = attr.string(default = ""),
        env_lib = attr.string(default = ""),
        env_path = attr.string(default = ""),
        _msvc_runtime = attr.label(
            default = "@config//:msvc_runtime",
            providers = [ConfigFlagInfo],
        ),
    ),
    provides = [CcToolchainConfigInfo],
)
