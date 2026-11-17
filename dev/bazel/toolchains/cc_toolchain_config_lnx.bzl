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

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")

load("@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "variable_with_value",
    "with_feature_set",
    "action_config",
    "tool",
)
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")
load("@onedal//dev/bazel/toolchains:cc_toolchain_config_common.bzl",
    "common",
    "all_compile_actions",
    "all_link_actions",
    "lto_index_actions",
    "COMMON_ATTRS",
)

def _impl(ctx):
    compiler_tools = common.compiler_tools(ctx.attr.cc_path, ctx.attr.dpcc_path)

    compile_actions = common.compile_action_configs(compiler_tools, [
        "default_compile_flags",
        "user_compile_flags",
        "unfiltered_compile_flags",
        "compiler_input_flags",
        "compiler_output_flags",
        "sysroot",
    ])

    cpp_link_executable_action = action_config(
        action_name = ACTION_NAMES.cpp_link_executable,
        implies = [
            "default_link_flags",
            "user_link_flags",
            "output_execpath_flags",
            "libraries_to_link",
            "runtime_library_search_directories",
            "library_search_directories",
            "linker_param_file",
            "force_pic_flags",
            "strip_debug_symbols",
            "sysroot",
            "default_dynamic_libraries",
        ],
        tools = compiler_tools,
    )

    cpp_link_nodeps_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
        implies = [
            "default_link_flags",
            "user_link_flags",
            "output_execpath_flags",
            "libraries_to_link",
            "library_search_directories",
            "linker_param_file",
            "force_pic_flags",
            "strip_debug_symbols",
            "sysroot",
        ],
        tools = compiler_tools,
    )

    cpp_link_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_dynamic_library,
        implies = [
            "shared_flag",
            "default_link_flags",
            "user_link_flags",
            "output_execpath_flags",
            "libraries_to_link",
            "library_search_directories",
            "linker_param_file",
            "force_pic_flags",
            "strip_debug_symbols",
            "sysroot",
            "default_dynamic_libraries",
        ],
        tools = compiler_tools,
    )

    cpp_link_static_library_action = common.cpp_link_static_library_action(ctx.attr.ar_path)

    strip_action = action_config(
        action_name = ACTION_NAMES.strip,
        flag_sets = [
            flag_set(
                flag_groups = [
                    flag_group(flags = ["-S", "-o", "%{output_file}"]),
                    # TODO: Add only if use GNU compiler stack
                    flag_group(
                        flags = [
                            "-R", ".gnu.switches.text.quote_paths'",
                            "-R", ".gnu.switches.text.bracket_paths",
                            "-R", ".gnu.switches.text.system_paths",
                            "-R", ".gnu.switches.text.cpp_defines",
                            "-R", ".gnu.switches.text.cpp_includes",
                            "-R", ".gnu.switches.text.cl_args",
                            "-R", ".gnu.switches.text.lipo_info",
                            "-R", ".gnu.switches.text.annotation",
                        ],
                    ),
                    flag_group(
                        flags = ["%{stripopts}"],
                        iterate_over = "stripopts",
                    ),
                    flag_group(flags = ["%{input_file}"]),
                ],
            ),
        ],
        tools = [
            tool(path = ctx.attr.strip_path)
        ],
    )

    action_configs = compile_actions + [
        cpp_link_executable_action,
        cpp_link_nodeps_dynamic_library_action,
        cpp_link_dynamic_library_action,
        cpp_link_static_library_action,
        strip_action,
    ]

    supports_pic_feature = feature(
        name = "supports_pic",
        enabled = True,
    )

    supports_start_end_lib_feature = feature(
        name = "supports_start_end_lib",
        enabled = True,
    )

    supports_dynamic_linker_feature = common.supports_dynamic_linker_feature()

    do_not_link_dynamic_dependencies_feature = common.do_not_link_dynamic_dependencies_feature()

    compiler_input_flags_feature = common.compiler_input_flags_feature()

    compiler_output_flags_feature = feature(
        name = "compiler_output_flags",
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [ "-S" ],
                        expand_if_available = "output_assembly_file",
                    ),
                    flag_group(
                        flags = [ "-E" ],
                        expand_if_available = "output_preprocess_file",
                    ),
                    flag_group(
                        flags = [ "-o", "%{output_file}" ],
                        expand_if_available = "output_file",
                    ),
                ],
            ),
        ],
    )

    strip_debug_symbols_feature = feature(
        name = "strip_debug_symbols",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = [ "-Wl,-S" ],
                        expand_if_available = "strip_debug_symbols",
                    ),
                ],
            ),
        ],
    )

    force_pic_flags_feature = feature(
        name = "force_pic_flags",
        flag_sets = [
            flag_set(
                actions = [ ACTION_NAMES.cpp_link_executable ],
                flag_groups = [
                    flag_group(
                        flags = [ "-pie" ],
                        expand_if_available = "force_pic",
                    ),
                ],
            ),
        ],
    )

    linker_param_file_feature = common.linker_param_file_feature()

    default_compile_flags_feature = common.default_compile_flags_feature(ctx.attr)

    default_link_flags_feature = common.default_link_flags_feature(
        ctx.attr.link_flags_cc,
        ctx.attr.link_flags_dpcc,
        ctx.attr.opt_link_flags,
    )

    sysroot_feature = feature(
        name = "sysroot",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.lto_backend,
                    ACTION_NAMES.clif_match,
                ] + all_link_actions + lto_index_actions,
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    user_compile_flags_feature = common.user_compile_flags_feature()

    user_link_flags_feature = common.user_link_flags_feature()

    default_dynamic_libraries_feature = common.default_dynamic_libraries_feature(
        ctx.attr.dynamic_link_libs,
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.no_canonical_system_headers_flags_cc,
                    ),
                ] if ctx.attr.no_canonical_system_headers_flags_cc else []),
                with_features = [with_feature_set(not_features = ["dpc++"])]
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.no_canonical_system_headers_flags_dpcc,
                    ),
                ] if ctx.attr.no_canonical_system_headers_flags_dpcc else []),
                with_features = [with_feature_set(features = ["dpc++"])]
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.deterministic_compile_flags,
                    ),
                ] if ctx.attr.deterministic_compile_flags else []),
            ),
        ],
    )

    library_search_directories_feature = feature(
        name = "library_search_directories",
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-L%{library_search_directories}"],
                        iterate_over = "library_search_directories",
                        expand_if_available = "library_search_directories",
                    ),
                ],
            ),
        ],
    )

    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
                ],
            ),
        ],
    )

    dpc_linking_pic_feature = feature(
        name = "dpc_linking_pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(flags = ["-fPIC"]),
                ],
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
        ],
    )

    preprocessor_defines_feature = feature(
        name = "preprocessor_defines",
        enabled = True,
        flag_sets = [
            flag_set(
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
                        flags = ["-D%{preprocessor_defines}"],
                        iterate_over = "preprocessor_defines",
                    ),
                ],
            ),
        ],
    )

    runtime_library_search_directories_feature = feature(
        name = "runtime_library_search_directories",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable] + lto_index_actions,
                flag_groups = [
                    flag_group(
                        iterate_over = "runtime_library_search_directories",
                        flag_groups = [
                            flag_group(
                                flags = [
                                    "-Wl,-rpath,$EXEC_ORIGIN/%{runtime_library_search_directories}",
                                ],
                                expand_if_true = "is_cc_test",
                            ),
                            flag_group(
                                flags = [
                                    "-Wl,-rpath,$ORIGIN/%{runtime_library_search_directories}",
                                ],
                                expand_if_false = "is_cc_test",
                            ),
                        ],
                        expand_if_available =
                            "runtime_library_search_directories",
                    ),
                ],
                with_features = [
                    with_feature_set(features = ["static_link_cpp_runtimes"]),
                ],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable] + lto_index_actions,
                flag_groups = [
                    flag_group(
                        iterate_over = "runtime_library_search_directories",
                        flag_groups = [
                            flag_group(
                                flags = [
                                    "-Wl,-rpath,$ORIGIN/%{runtime_library_search_directories}",
                                ],
                            ),
                        ],
                        expand_if_available =
                            "runtime_library_search_directories",
                    ),
                ],
                with_features = [
                    with_feature_set(
                        not_features = ["static_link_cpp_runtimes"],
                    ),
                ],
            ),
        ],
    )

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
                flag_groups = [flag_group(flags = ["-shared"])],
            ),
        ],
    )

    random_seed_feature = feature(
        name = "random_seed",
        enabled = ctx.attr.supports_random_seed,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-frandom-seed=%{output_file}"],
                        expand_if_available = "output_file",
                    ),
                ],
            ),
        ],
    )

    includes_feature = feature(
        name = "includes",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-include", "%{includes}"],
                        iterate_over = "includes",
                        expand_if_available = "includes",
                    ),
                ],
            ),
        ],
    )

    include_paths_feature = feature(
        name = "include_paths",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.linkstamp_compile,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.clif_match,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-iquote", "%{quote_include_paths}"],
                        iterate_over = "quote_include_paths",
                    ),
                    flag_group(
                        flags = ["-I%{include_paths}"],
                        iterate_over = "include_paths",
                    ),
                    flag_group(
                        flags = ["-isystem", "%{system_include_paths}"],
                        iterate_over = "system_include_paths",
                    ),
                ],
            ),
        ],
    )

    libraries_to_link_feature = feature(
        name = "libraries_to_link",
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                flags = ["-Wl,--start-lib"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flags = ["-Wl,-whole-archive"],
                                expand_if_true =
                                    "libraries_to_link.is_whole_archive",
                            ),
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
                            flag_group(
                                flags = ["-Wl,-no-whole-archive"],
                                expand_if_true = "libraries_to_link.is_whole_archive",
                            ),
                            flag_group(
                                flags = ["-Wl,--end-lib"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                flags = ["%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "interface_library",
                                ),
                            ),
                            flag_group(
                                flags = ["-l%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "dynamic_library",
                                ),
                            ),
                            flag_group(
                                flags = ["-l:%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "versioned_dynamic_library",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                ],
                with_features = [
                    with_feature_set(
                        not_features = ["do_not_link_dynamic_dependencies"],
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,@%{thinlto_param_file}"],
                        expand_if_true = "thinlto_param_file",
                    ),
                ],
            ),
        ],
    )

    archiver_flags_feature = feature(
        name = "archiver_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(flags = ["rcsD"]),
                    flag_group(
                        flags = ["%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
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
                    ),
                ],
            ),
        ],
    )

    dependency_file_feature = feature(
        name = "dependency_file",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.objc_compile,
                    ACTION_NAMES.objcpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-MD", "-MF", "%{dependency_file}"],
                        expand_if_available = "dependency_file",
                    ),
                ],
            ),
        ],
    )

    output_execpath_flags_feature = feature(
        name = "output_execpath_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-o", "%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    features = [common.no_legacy_features_feature()]
    features.extend(common.marker_features(ctx.attr.dpcc_path))
    features.extend([
        supports_pic_feature,
        supports_dynamic_linker_feature,
        do_not_link_dynamic_dependencies_feature,
        sysroot_feature,
    ])

    # Compilation
    features.extend([
        default_compile_flags_feature,
        force_pic_flags_feature,
        pic_feature,
    ])
    features.extend(common.cpu_opt_features(ctx.attr.cpu_flags_cc, ctx.attr.cpu_flags_dpcc))
    features.extend([
        user_compile_flags_feature,
        preprocessor_defines_feature,
        includes_feature,
        include_paths_feature,
        unfiltered_compile_flags_feature,
        dependency_file_feature,
        random_seed_feature,
        compiler_input_flags_feature,
        compiler_output_flags_feature,
    ])

    # Dynamic linking
    features.extend([
        strip_debug_symbols_feature,
        shared_flag_feature,
        output_execpath_flags_feature,
        default_link_flags_feature,
        dpc_linking_pic_feature,
        library_search_directories_feature,
        runtime_library_search_directories_feature,
        libraries_to_link_feature,
        user_link_flags_feature,
        default_dynamic_libraries_feature,
    ])

    # Static linking
    features.extend([
        archiver_flags_feature,
        linker_param_file_feature,
    ])
    if ctx.attr.supports_start_end_lib:
        features.append(supports_start_end_lib_feature)

    return common.create_config_info(ctx, features, action_configs)

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = dict(
        COMMON_ATTRS,
        strip_path = attr.string(mandatory = True),
        no_canonical_system_headers_flags_cc = attr.string_list(),
        no_canonical_system_headers_flags_dpcc = attr.string_list(),
        supports_start_end_lib = attr.bool(),
        supports_random_seed = attr.bool(),
    ),
    provides = [CcToolchainConfigInfo],
)
