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

# Custom cc_toolchain_config for Intel oneAPI icx on Windows.
#
# icx.exe runs in its native clang-cl driver mode (MSVC-compatible syntax),
# matching dev/make/compiler_definitions/icx.mkl.32e.mk. All flags below use
# MSVC-style spellings (`/I`, `/imsvc`, `/Fo`, `/Qstd:c++17`, `-MD`, …) so
# icx accepts them without a driver-mode flip. The major differences vs.
# cc_toolchain_config_lnx.bzl:
#
#   * Flags use `/FLAG`/`-FLAG` clang-cl syntax, not gcc-style.
#   * .obj / .lib / .dll output extensions (set via artifact_name_pattern).
#   * No -fPIC — Windows is always position-independent.
#   * Static archives produced via `lib.exe` (or equivalent), not `ar`.
#   * Link driven through icx with `-link` forwarding to lld-link.exe.

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")

load("@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "env_entry",
    "env_set",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
    "action_config",
    "tool",
    "artifact_name_pattern",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl", "CcToolchainConfigInfo")

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
        with_features = [with_feature_set(not_features = ["dpc++"])],
    )
    dpcc_tool = tool(
        path = ctx.attr.dpcc_path,
        with_features = [with_feature_set(features = ["dpc++"])],
    )
    cc_link_tool = tool(
        path = ctx.attr.cc_link_path,
        with_features = [with_feature_set(not_features = ["dpc++"])],
    )
    dpcc_link_tool = tool(
        path = ctx.attr.dpcc_link_path,
        with_features = [with_feature_set(features = ["dpc++"])],
    )

    # --- action configs -----------------------------------------------------

    common_compile_implies = [
        "default_compile_flags",
        "user_compile_flags",
        "unfiltered_compile_flags",
        "compiler_input_flags",
        "compiler_output_flags",
    ]

    assemble_action = action_config(
        action_name = ACTION_NAMES.assemble,
        implies = common_compile_implies,
        tools = [cc_tool, dpcc_tool],
    )
    preprocess_assemble_action = action_config(
        action_name = ACTION_NAMES.preprocess_assemble,
        implies = common_compile_implies,
        tools = [cc_tool, dpcc_tool],
    )
    c_compile_action = action_config(
        action_name = ACTION_NAMES.c_compile,
        implies = common_compile_implies,
        tools = [cc_tool, dpcc_tool],
    )
    cpp_compile_action = action_config(
        action_name = ACTION_NAMES.cpp_compile,
        implies = common_compile_implies,
        tools = [cc_tool, dpcc_tool],
    )
    cpp_header_parsing_action = action_config(
        action_name = ACTION_NAMES.cpp_header_parsing,
        implies = common_compile_implies,
        tools = [cc_tool, dpcc_tool],
    )

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
        tools = [cc_link_tool, dpcc_link_tool],
    )
    cpp_link_nodeps_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_nodeps_dynamic_library,
        implies = common_link_implies + ["shared_flag"],
        tools = [cc_link_tool, dpcc_link_tool],
    )
    cpp_link_dynamic_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_dynamic_library,
        implies = common_link_implies + ["shared_flag", "default_dynamic_libraries"],
        tools = [cc_link_tool, dpcc_link_tool],
    )
    cpp_link_static_library_action = action_config(
        action_name = ACTION_NAMES.cpp_link_static_library,
        implies = ["archiver_flags", "linker_param_file"],
        tools = [tool(path = ctx.attr.ar_path)],
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
    ]

    # --- features -----------------------------------------------------------

    dpc_feature = feature(name = "dpc++")
    cxx11_feature = feature(name = "c++11")
    cxx14_feature = feature(name = "c++14")
    cxx17_feature = feature(name = "c++17")
    pedantic_feature = feature(name = "pedantic")
    dbg_feature = feature(name = "dbg")
    opt_feature = feature(name = "opt")
    # Windows PE/COFF objects have no PIC/non-PIC distinction, so we do not
    # register a `supports_pic` feature. Matches rules_cc's MSVC auto-config
    # and keeps `cc_common.compile` from emitting both variants (which the
    # cc_module rule in dev/bazel/cc.bzl explicitly rejects).
    supports_dynamic_linker_feature = feature(
        name = "supports_dynamic_linker", enabled = True,
    )
    do_not_link_dynamic_dependencies_feature = feature(
        name = "do_not_link_dynamic_dependencies", enabled = False,
    )
    no_legacy_features_feature = feature(name = "no_legacy_features", enabled = True)

    # Inject Windows shell env vars into every compile and link action so
    # icx (via `INCLUDE`) and lld-link (via `LIB`) find MSVC, Windows SDK,
    # and oneAPI headers/libs without relying on the user's inherited
    # shell — the Makefile relies on these same vars set by setvars.bat.
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

    compiler_input_flags_feature = feature(
        name = "compiler_input_flags",
        flag_sets = [flag_set(
            actions = all_compile_actions,
            flag_groups = [flag_group(
                flags = ["-c", "%{source_file}"],
                expand_if_available = "source_file",
            )],
        )],
    )

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

    linker_param_file_feature = feature(
        name = "linker_param_file",
        flag_sets = [flag_set(
            actions = all_link_actions + [ACTION_NAMES.cpp_link_static_library],
            flag_groups = [flag_group(
                flags = ["@%{linker_param_file}"],
                expand_if_available = "linker_param_file",
            )],
        )],
    )

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([flag_group(flags = ctx.attr.compile_flags_cc)]
                               if ctx.attr.compile_flags_cc else []),
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([flag_group(flags = ctx.attr.compile_flags_dpcc)]
                               if ctx.attr.compile_flags_dpcc else []),
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([flag_group(flags = ctx.attr.dbg_compile_flags)]
                               if ctx.attr.dbg_compile_flags else []),
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([flag_group(flags = ctx.attr.opt_compile_flags)]
                               if ctx.attr.opt_compile_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["/Qstd:c++11"])],
                with_features = [with_feature_set(features = ["c++11"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["/Qstd:c++14"])],
                with_features = [with_feature_set(features = ["c++14"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = ["/Qstd:c++17"])],
                with_features = [with_feature_set(features = ["c++17"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([flag_group(flags = ctx.attr.compile_flags_pedantic_cc)]
                               if ctx.attr.compile_flags_pedantic_cc else []),
                with_features = [with_feature_set(features = ["pedantic"],
                                                  not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([flag_group(flags = ctx.attr.compile_flags_pedantic_dpcc)]
                               if ctx.attr.compile_flags_pedantic_dpcc else []),
                with_features = [with_feature_set(features = ["dpc++", "pedantic"])],
            ),
            flag_set(
                actions = all_cpp_compile_actions + [ACTION_NAMES.lto_backend],
                flag_groups = ([flag_group(flags = ctx.attr.cxx_flags)]
                               if ctx.attr.cxx_flags else []),
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([flag_group(flags = ctx.attr.link_flags_cc)]
                               if ctx.attr.link_flags_cc else []),
                with_features = [with_feature_set(not_features = ["dpc++"])],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([flag_group(flags = ctx.attr.link_flags_dpcc)]
                               if ctx.attr.link_flags_dpcc else []),
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
            flag_set(
                actions = all_link_actions + lto_index_actions,
                flag_groups = ([flag_group(flags = ctx.attr.opt_link_flags)]
                               if ctx.attr.opt_link_flags else []),
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        enabled = True,
        flag_sets = [flag_set(
            actions = all_compile_actions,
            flag_groups = [flag_group(
                flags = ["%{user_compile_flags}"],
                iterate_over = "user_compile_flags",
                expand_if_available = "user_compile_flags",
            )],
        )],
    )

    user_link_flags_feature = feature(
        name = "user_link_flags",
        flag_sets = [flag_set(
            actions = all_link_actions + lto_index_actions,
            flag_groups = [flag_group(
                flags = ["%{user_link_flags}"],
                iterate_over = "user_link_flags",
                expand_if_available = "user_link_flags",
            )],
        )],
    )

    default_dynamic_libraries_feature = feature(
        name = "default_dynamic_libraries",
        flag_sets = [flag_set(
            actions = all_link_actions + lto_index_actions,
            flag_groups = ([flag_group(flags = ctx.attr.dynamic_link_libs)]
                           if ctx.attr.dynamic_link_libs else []),
            with_features = [with_feature_set(
                not_features = ["do_not_link_dynamic_dependencies"],
            )],
        )],
    )

    # DPC++ links go through the icx clang-cl driver so it can add SYCL device
    # runtime libraries. Keep objects/libraries before `/link`, then put
    # linker-native MSVC options after it; otherwise icx can treat `/DLL` and
    # `/OUT:` as driver inputs and invoke link.exe as an executable link, which
    # fails with LNK1561.
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

    # Windows DLLs live alongside their import .lib. clang-cl forwards
    # library search paths to lld-link when they start with `-libpath:`
    # (lowercase, no slash prefix that confuses its input-file detector).
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

    # Force-include a header (`-include foo.h` in gcc; `/FI foo.h` in clang-cl).
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

    # clang-cl accepts `-I<dir>` / `/I <dir>` for user includes and `/imsvc <dir>`
    # for system-include paths (the clang-cl equivalent of `-isystem`). Quote
    # includes have no distinct syntax in MSVC-driver mode, so they collapse
    # into plain `/I`.
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

    # Libraries-to-link uses clang-style -l wiring; lld-link accepts it when
    # invoked via the icx/icpx driver. Note: we do NOT emit -Wl,--start-lib /
    # --whole-archive because those are ld/GNU-only. Static libraries end up
    # as plain positional .lib paths, which MSVC-style linkers handle.
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
                        # MSVC-style: dynamic libraries link via their import
                        # .lib name as a positional argument (no `-l`).
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

    # On Windows the archiver is `lib.exe`; it accepts `/OUT:<output>` and
    # takes object files as positional args. We invoke it through lld-link
    # --lib-compat syntax (llvm-lib / lib.exe both understand /OUT:).
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

    # icx in clang-cl mode emits dependency info via clang-style flags passed
    # through `/clang:` so they don't collide with MSVC's `-MD` (which in
    # clang-cl means "link against msvcrt", not "generate .d file").
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
        flag_sets = [flag_set(
            actions = all_link_actions + lto_index_actions,
            flag_groups = [flag_group(
                flags = ["/OUT:%{output_execpath}"],
                expand_if_available = "output_execpath",
            )],
        )],
    )

    # Non-DPC links use lld-link.exe directly (see _find_tools_icx in
    # cc_toolchain_win.bzl), so they can take linker-native `/DLL` directly.
    # DPC++ links go through the icx driver; `dpc_linker_mode` puts `/link`
    # before this flag so `/DLL` is forwarded to the underlying linker instead
    # of being interpreted as a driver input.
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
                flag_groups = [flag_group(flags = ["/DLL"])],
                with_features = [with_feature_set(features = ["dpc++"])],
            ),
        ],
    )

    # --- assemble feature list ---------------------------------------------

    features = [
        no_legacy_features_feature,
        msvc_env_feature,
        dpc_feature,
        cxx11_feature,
        cxx14_feature,
        cxx17_feature,
        pedantic_feature,
        dbg_feature,
        opt_feature,
        supports_dynamic_linker_feature,
        do_not_link_dynamic_dependencies_feature,
        default_compile_flags_feature,
    ]

    for cpu_id, flag_list in ctx.attr.cpu_flags_cc.items():
        cpu_opt_feature = feature(
            name = "{}_flags".format(cpu_id),
            flag_sets = [
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [flag_group(flags = flag_list)],
                    with_features = [with_feature_set(not_features = ["dpc++"])],
                ),
                flag_set(
                    actions = all_compile_actions,
                    flag_groups = [flag_group(
                        flags = ctx.attr.cpu_flags_dpcc[cpu_id],
                    )],
                    with_features = [with_feature_set(features = ["dpc++"])],
                ),
            ],
        )
        features.append(cpu_opt_feature)

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
        library_search_directories_feature,
        libraries_to_link_feature,
        dpc_linker_mode_feature,
        shared_flag_feature,
        output_execpath_flags_feature,
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
            # Bazel only accepts `.ifso`, `.tbd`, `.if.lib`, or `.lib`.
            # The release-staging rule renames these to the conventional
            # `_dll.lib` Windows-Make naming (see release.bzl).
            extension = ".if.lib",
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        artifact_name_patterns = artifact_name_patterns,
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
        # `INCLUDE` / `LIB` / `PATH` values captured at repo-configure time
        # so bazel-sandboxed compile and link actions can find MSVC + Windows
        # SDK + oneAPI headers/libs without the shell env being inherited.
        "env_include": attr.string(default = ""),
        "env_lib": attr.string(default = ""),
        "env_path": attr.string(default = ""),
    },
    provides = [CcToolchainConfigInfo],
)
