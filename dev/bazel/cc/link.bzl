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

load("@onedal//dev/bazel:utils.bzl",
    "utils",
    "paths",
    "sets",
)
load("@onedal//dev/bazel/toolchains:action_names.bzl",
    "CPP_MERGE_STATIC_LIBRARIES"
)
load("@onedal//dev/bazel/cc:common.bzl",
    onedal_cc_common = "common",
)

def _filter_user_link_flags(feature_configuration, user_link_flags):
    strip_dynamic_libraries_from_user_link_flags = cc_common.is_enabled(
        feature_configuration = feature_configuration,
        feature_name = "do_not_link_dynamic_dependencies",
    )
    if strip_dynamic_libraries_from_user_link_flags:
        filtered_flags = []
        for flag in user_link_flags:
            if not flag.startswith("-l"):
                filtered_flags.append(flag)
        return filtered_flags
    return user_link_flags

def _merge_static_libs(filename, actions, cc_toolchain,
                       feature_configuration, static_libs):
    output_file = actions.declare_file(filename)
    merger_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = CPP_MERGE_STATIC_LIBRARIES,
    )
    merger_variables = cc_common.create_link_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        is_using_linker = False,
    )
    command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = CPP_MERGE_STATIC_LIBRARIES,
        variables = merger_variables,
    )
    args = actions.args()
    args.add_all(command_line)
    args.add(output_file)
    args.add_all(static_libs)
    env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = CPP_MERGE_STATIC_LIBRARIES,
        variables = merger_variables,
    )
    actions.run(
        executable = merger_path,
        arguments = [args],
        env = env,
        inputs = depset(
            direct = static_libs,
            transitive = [
                cc_toolchain.all_files,
            ],
        ),
        outputs = [output_file],
    )
    return output_file

def _static(owner, name, actions, cc_toolchain,
            feature_configuration, linking_contexts):
    unpacked_linking_context = onedal_cc_common.unpack_linking_contexts(linking_contexts)
    if (unpacked_linking_context.objects and
        unpacked_linking_context.pic_objects):
        utils.warn("Static library {} contains mix of PIC and non-PIC code".format(name))
    all_objects = depset(unpacked_linking_context.pic_objects +
                         unpacked_linking_context.objects)
    compilation_outputs = cc_common.create_compilation_outputs(
        objects = all_objects,
        pic_objects = all_objects,
    )
    _, linking_outputs = cc_common.create_linking_context_from_compilation_outputs(
        name = name + ("_no_deps" if unpacked_linking_context.static_libraries else ""),
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        compilation_outputs = compilation_outputs,
        linking_contexts = linking_contexts,
        disallow_dynamic_library = True,
    )
    if not linking_outputs.library_to_link:
        return utils.warn("'{}' static library does not contain any " +
                          "object file".format(name))
    static_lib = (linking_outputs.library_to_link.static_library or
                  linking_outputs.library_to_link.pic_static_library)
    if unpacked_linking_context.static_libraries:
        static_lib = _merge_static_libs(
            filename = utils.remove_substring(static_lib.basename, "_no_deps"),
            actions = actions,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            static_libs = [ static_lib ] + unpacked_linking_context.static_libraries,
        )
    static_lib_to_link = cc_common.create_library_to_link(
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        static_library = static_lib,
        pic_static_library = static_lib,
    )
    linker_input = cc_common.create_linker_input(
        owner = owner,
        libraries = depset([static_lib_to_link] +
                           unpacked_linking_context.dynamic_libraries_to_link),
        user_link_flags = depset(unpacked_linking_context.user_link_flags),
    )
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([ linker_input ]),
    )
    return linking_context, static_lib

def _link(owner, name, actions, cc_toolchain,
          feature_configuration, linking_contexts,
          def_file=None, is_executable=False, user_link_flags=[]):
    unpacked_linking_context = onedal_cc_common.unpack_linking_contexts(linking_contexts)
    if not is_executable and unpacked_linking_context.objects:
        fail("Dynamic library {} contains non-PIC object files: {}".format(
            name, unpacked_linking_context.objects))
    all_objects = depset(unpacked_linking_context.pic_objects +
                         unpacked_linking_context.objects)
    compilation_outputs = cc_common.create_compilation_outputs(
        objects = all_objects,
        pic_objects = all_objects,
    )
    unpacked_user_link_flags = _filter_user_link_flags(
        feature_configuration,
        unpacked_linking_context.user_link_flags
    )
    # Merge user_link_flags from both linking context and function parameters
    all_user_link_flags = unpacked_user_link_flags + user_link_flags
    linker_input = cc_common.create_linker_input(
        owner = owner,
        libraries = depset(unpacked_linking_context.libraries_to_link),
        user_link_flags = depset(all_user_link_flags),
    )
    # TODO: Pass compilations outputs via linking contexts
    #       Individual linking context for each library tag
    #       This will help optimize link time via --start-lib/--end-lib
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([linker_input]),
    )
    linking_outputs = cc_common.link(
        name = name,
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        compilation_outputs = compilation_outputs,
        linking_contexts = [linking_context],
        output_type = "executable" if is_executable else "dynamic_library",
        link_deps_statically = True,
        user_link_flags = (
            (["@" + def_file.path] if def_file else []) +
            user_link_flags
        ),
        additional_inputs = [def_file] if def_file else [],
    )
    return unpacked_linking_context, linking_outputs

def _dynamic(owner, name, actions, cc_toolchain,
             feature_configuration, linking_contexts,
             def_file=None, user_link_flags=[]):
    unpacked_linking_context, linking_outputs = _link(
        owner, name, actions, cc_toolchain,
        feature_configuration, linking_contexts,
        def_file,
        user_link_flags=user_link_flags,
    )
    library_to_link = linking_outputs.library_to_link
    if not (library_to_link and library_to_link.resolved_symlink_dynamic_library):
        return utils.warn("'{}' dynamic library does not contain any " +
                          "object file".format(name))
    # TODO: Handle interface dynamic library on Windows
    dynamic_lib = library_to_link.resolved_symlink_dynamic_library
    dynamic_lib_to_link = cc_common.create_library_to_link(
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        dynamic_library = dynamic_lib,
    )
    linker_input = cc_common.create_linker_input(
        owner = owner,
        libraries = depset([dynamic_lib_to_link] +
                           unpacked_linking_context.dynamic_libraries_to_link),
        user_link_flags = depset(unpacked_linking_context.user_link_flags),
    )
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([ linker_input ]),
    )
    return linking_context, dynamic_lib

def _executable(owner, name, actions, cc_toolchain,
                feature_configuration, linking_contexts,
                user_link_flags=[]):
    _, linking_outputs = _link(
        owner, name, actions, cc_toolchain,
        feature_configuration, linking_contexts,
        is_executable = True,
        user_link_flags = user_link_flags,
    )
    if not linking_outputs.executable:
        return utils.warn("'{}' executable does not contain any " +
                          "object file".format(name))
    return linking_outputs.executable

link = struct(
    static = _static,
    dynamic = _dynamic,
    executable = _executable,
)
