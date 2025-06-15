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

load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
    "action_config",
    "tool",
    "artifact_name_pattern",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@onedal//dev/bazel/toolchains:action_names.bzl", "CPP_MERGE_STATIC_LIBRARIES")

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

preprocessor_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.clif_match,
]

codegen_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_module_codegen,
    ACTION_NAMES.lto_backend,
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

def _impl(ctx):
    cc_tool = tool(
        path = ctx.attr.cc_path,
        with_features = [
            with_feature_set(not_features = ["dpc++"])
        ]
    )

    dpcc_tool = tool(
        path = ctx.attr.dpcc_path,
        with_features = [
            with_feature_set(features = ["dpc++"]),
        ],
    )

    cc_link_tool = tool(
        path = ctx.attr.cc_link_path,
        with_features = [
            with_feature_set(not_features = ["dpc++"])
        ]
    )

    dpcc_link_tool = tool(
        path = ctx.attr.dpcc_link_path,
        with_features = [
            with_feature_set(features = ["dpc++"]),
        ],
    )

    assemble_action = action_config(
        action_name = ACTION_NAMES.assemble,
        implies = [
            "default_compile_flags",
            "user_compile_flags",
            "unfiltered_compile_flags",
            "compiler_input_flags",
            "compiler_output_flags",
            "sysroot",
        ],
        tools = [ cc_tool, dpcc_tool ],
    )

    preprocess_assemble_action = action_config(
        action_name = ACTION_NAMES.preprocess_assemble,
        implies = [
            "default_compile_flags",
            "user_compile_flags",
            "unfiltered_compile_flags",
            "compiler_input_flags",
            "compiler_output_flags",
            "sysroot",
        ],
        tools = [ cc_tool, dpcc_tool ],
    )

    c_compile_action = action_config(
        action_name = ACTION_NAMES.c_compile,
        implies = [
            "default_compile_flags",
            "user_compile_flags",
            "unfiltered_compile_flags",
            "compiler_input_flags",
            "compiler_output_flags",
            "sysroot",
        ],
        tools = [ cc_tool, dpcc_tool ],
    )

    cpp_compile_action = action_config(
        action_name = ACTION_NAMES.cpp_compile,
        implies = [
            "default_compile_flags",
            "user_compile_flags",
            "unfiltered_compile_flags",
            "compiler_input_flags",
            "compiler_output_flags",
            "sysroot",
        ],
        tools = [ cc_tool, dpcc_tool ],
    )

    cpp_header_parsing_action = action_config(
        action_name = ACTION_NAMES.cpp_header_parsing,
        implies = [
            "default_compile_flags",
            "user_compile_flags",
            "unfiltered_compile_flags",
            "compiler_input_flags",
            "compiler_output_flags",
            "sysroot",
        ],
        tools = [ cc_tool, dpcc_tool ],
    )

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
        tools = [ cc_link_tool, dpcc_link_tool ],
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
        tools = [ cc_link_tool, dpcc_link_tool ],
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
        tools = [ cc_link_tool, dpcc_link_tool ],
    )

    cpp_link_static_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_static_library,
        implies = [
            "archiver_flags",
            "linker_param_file",
        ],
        tools = [
            tool(path = ctx.attr.lib_path)
        ],
    )

    cpp_merge_static_libraries_action = action_config(
        action_name = CPP_MERGE_STATIC_LIBRARIES,
        tools = [
            tool(path = ctx.attr.lib_merge_path)
        ],
    )

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

    action_configs = [
        assemble_action,
        preprocess_assemble_action,
        c_compile_action,
        cpp_compile_action,
        cpp_header_parsing_action,
        cpp_link_executable_action,
        cpp_link_nodeps_dynamic_library_action,
        cpp_link_dynamic_library_action,
        cpp_link_static_library_action,
        cpp_merge_static_libraries_action,
        strip_action,
    ]

    dpc_feature = feature(
        name = "dpc++",
    )

    cxx11_feature = feature(
        name = "c++11",
    )

    cxx14_feature = feature(
        name = "c++14",
    )

    cxx17_feature = feature(
        name = "c++17",
    )

    pedantic_feature = feature(
        name = "pedantic",
    )

    dbg_feature = feature(
        name = "dbg"
    )

    opt_feature = feature(
        name = "opt"
    )

    supports_pic_feature = feature(
        name = "supports_pic",
        enabled = True,
    )

    supports_start_end_lib_feature = feature(
        name = "supports_start_end_lib",
        enabled = True,
    )

    supports_dynamic_linker_feature = feature(
        name = "supports_dynamic_linker",
        enabled = True
    )

    do_not_link_dynamic_dependencies_feature = feature(
        name = "do_not_link_dynamic_dependencies",
        enabled = False
    )

    compiler_input_flags_feature = feature(
        name = "compiler_input_flags",
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [ "-c", "%{source_file}" ],
                        expand_if_available = "source_file",
                    ),
                ],
            ),
        ],
    )

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

    linker_param_file_feature = feature(
        name = "linker_param_file",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = [ "@%{linker_param_file}" ],
                        expand_if_available = "linker_param_file",
                    ),
                ],
            ),
        ],
    )

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.compile_flags_cc,
                    ),
                ] if ctx.attr.compile_flags_cc else []),
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.compile_flags_dpcc,
                    ),
                ] if ctx.attr.compile_flags_dpcc else []),
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.dbg_compile_flags,
                    ),
                ] if ctx.attr.dbg_compile_flags else []),
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.opt_compile_flags,
                    ),
                ] if ctx.attr.opt_compile_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [ "-std=c++11" ],
                    ),
                ],
                with_features = [with_feature_set(features = ["c++11"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [ "-std=c++14" ],
                    ),
                ],
                with_features = [with_feature_set(features = ["c++14"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [ "-std=c++17" ],
                    ),
                ],
                with_features = [with_feature_set(features = ["c++17"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.compile_flags_pedantic_cc,
                    ),
                ] if ctx.attr.compile_flags_pedantic_cc else [],
                with_features = [with_feature_set(features = ["pedantic"],
                                                  not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.compile_flags_pedantic_dpcc,
                    ),
                ] if ctx.attr.compile_flags_pedantic_dpcc else [],
                with_features = [with_feature_set(features = ["dpc++", "pedantic"])],
            ),
            flag_set(
                actions = all_cpp_compile_actions + [ACTION_NAMES.lto_backend],
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.cxx_flags,
                    ),
                ] if ctx.attr.cxx_flags else []),
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.link_flags_cc,
                    ),
                ] if ctx.attr.link_flags_cc else []),
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.link_flags_dpcc,
                    ),
                ] if ctx.attr.link_flags_dpcc else []),
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([
                    flag_group(
                        flags = ctx.attr.opt_link_flags,
                    ),
                ] if ctx.attr.opt_link_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
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

    user_compile_flags_feature = feature(
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

    user_link_flags_feature = feature(
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

    default_dynamic_libraries_feature = feature(
        name = "default_dynamic_libraries",
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = (
                    [flag_group(flags = ctx.attr.dynamic_link_libs)]
                    if ctx.attr.dynamic_link_libs else []
                ),
                with_features = [
                    with_feature_set(
                        not_features = ["do_not_link_dynamic_dependencies"],
                    ),
                ],
            ),
        ],
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
                actions = all_link_actions + lto_index_actions,
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
                actions = all_link_actions + lto_index_actions,
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
                flag_group(flags = ["/OUT:%{output_execpath}"]),
                # REMOVE THIS ENTIRE flag_group:
                # flag_group(
                #     flags = ["%{output_execpath}"],
                #     expand_if_available = "output_execpath",
                # ),
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

    no_legacy_features_feature = feature(
        name = "no_legacy_features",
        enabled = True,
    )

    features = []
    features.append(no_legacy_features_feature)
    features.append(dpc_feature)
    features.append(cxx11_feature)
    features.append(cxx14_feature)
    features.append(cxx17_feature)
    features.append(pedantic_feature)
    features.append(dbg_feature)
    features.append(opt_feature)
    features.append(supports_pic_feature)
    features.append(supports_dynamic_linker_feature)
    features.append(do_not_link_dynamic_dependencies_feature)
    features.append(sysroot_feature)

    # Compilation
    features.append(default_compile_flags_feature)
    features.append(force_pic_flags_feature)
    features.append(pic_feature)
    for cpu_id, flag_list in ctx.attr.cpu_flags_cc.items():
        cpu_opt_feature = feature(
            name = "{}_flags".format(cpu_id),
            flag_sets = [
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [ flag_group(flags = flag_list) ],
                    with_features = [with_feature_set(not_features = ["dpc++"])],
                ),
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [ flag_group(flags = ctx.attr.cpu_flags_dpcc[cpu_id]) ],
                    with_features = [with_feature_set(features = ["dpc++"])],
                ),
            ],
        )
        features.append(cpu_opt_feature)
    features.append(user_compile_flags_feature)
    features.append(preprocessor_defines_feature)
    features.append(includes_feature)
    features.append(include_paths_feature)
    features.append(unfiltered_compile_flags_feature)
    features.append(dependency_file_feature)
    features.append(random_seed_feature)
    features.append(compiler_input_flags_feature)
    features.append(compiler_output_flags_feature)

    # Dynamic linking
    features.append(strip_debug_symbols_feature)
    features.append(shared_flag_feature)
    features.append(output_execpath_flags_feature)
    features.append(default_link_flags_feature)
    features.append(dpc_linking_pic_feature)
    features.append(library_search_directories_feature)
    features.append(runtime_library_search_directories_feature)
    features.append(libraries_to_link_feature)
    features.append(user_link_flags_feature)
    features.append(default_dynamic_libraries_feature)

    # Static linking
    features.append(archiver_flags_feature)
    features.append(linker_param_file_feature)
    if ctx.attr.supports_start_end_lib:
        features.append(supports_start_end_lib_feature)

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
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
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
        "cc_link_path": attr.string(mandatory = True),
        "dpcc_link_path": attr.string(mandatory = True),
        "lib_path": attr.string(mandatory = True),
        "lib_merge_path": attr.string(mandatory = True),
        "strip_path": attr.string(mandatory = True),
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
        "no_canonical_system_headers_flags_cc": attr.string_list(),
        "no_canonical_system_headers_flags_dpcc": attr.string_list(),
        "deterministic_compile_flags": attr.string_list(),
        "supports_start_end_lib": attr.bool(),
        "supports_random_seed": attr.bool(),
        "cpu_flags_cc": attr.string_list_dict(),
        "cpu_flags_dpcc": attr.string_list_dict(),
    },
    provides = [CcToolchainConfigInfo],
)
