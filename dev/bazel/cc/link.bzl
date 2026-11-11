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

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")

load("@onedal//dev/bazel:utils.bzl",
    "utils",
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
                       feature_configuration, static_libs, is_windows = False):
    """Archive whole static libraries into a single library.

    cc_common would add each input archive as a member, so the toolchain's archiver runs directly;
    GNU `ar` reads an MRI script from stdin.

    Args:
        filename: basename of the library to produce.
        actions: the rule context's `actions`.
        cc_toolchain: the resolved `CcToolchainInfo`.
        feature_configuration: the configured features, used to look up the
                               archiver.
        static_libs: static libraries to merge, plus loose object files on
                     Windows.
        is_windows: whether the target platform is Windows.

    Returns:
        The merged library `File`.
    """
    output_file = actions.declare_file(filename)
    archiver_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_static_library,
    )
    if is_windows:
        args = actions.args()
        args.use_param_file("@%s", use_always = True)
        args.set_param_file_format("multiline")
        args.add("/NOLOGO")
        args.add("/OUT:" + output_file.path)
        args.add_all(static_libs)
        actions.run(
            executable = archiver_path,
            arguments = [args],
            inputs = static_libs,
            outputs = [output_file],
            mnemonic = "MergeStaticLibraries",
            use_default_shell_env = True,
        )
        return output_file
    mri_script = actions.declare_file(filename + ".mri")
    actions.write(
        output = mri_script,
        content = "\n".join(
            ["CREATE " + output_file.path] +
            ["ADDLIB " + lib.path for lib in static_libs] +
            ["SAVE", ""]
        ),
    )
    actions.run_shell(
        command = '"{}" -M < "{}"'.format(archiver_path, mri_script.path),
        inputs = depset(
            direct = static_libs + [mri_script],
            transitive = [
                cc_toolchain.all_files,
            ],
        ),
        outputs = [output_file],
        mnemonic = "MergeStaticLibraries",
        progress_message = "Merging static libraries into %{output}",
    )
    return output_file

def _static(owner, name, actions, cc_toolchain,
            feature_configuration, linking_contexts, is_windows = False):
    unpacked = onedal_cc_common.unpack_linking_contexts(linking_contexts)
    if unpacked.objects and unpacked.pic_objects:
        utils.warn("Static library {} contains mix of PIC and non-PIC code".format(name))
    objects = unpacked.pic_objects + unpacked.objects
    archives = unpacked.static_libraries
    filename = "{}.lib".format(name) if is_windows else "lib{}.a".format(name)
    if is_windows:
        # `lib` takes objects and archives together. Without any input it
        # writes nothing, so a library whose modules have no objects is empty.
        if objects or archives:
            static_lib = _merge_static_libs(
                filename = filename,
                actions = actions,
                cc_toolchain = cc_toolchain,
                feature_configuration = feature_configuration,
                static_libs = objects + archives,
                is_windows = True,
            )
        else:
            static_lib = actions.declare_file(filename)
            actions.write(output = static_lib, content = "")
    else:
        if not objects and not archives:
            fail("'{}' static library does not contain any object file".format(name))
        static_lib = None
        if objects:
            # The toolchain's archiving action; `_no_deps` names the archive of
            # the own objects when dependency archives are merged in afterwards.
            all_objects = depset(objects)
            _, linking_outputs = cc_common.create_linking_context_from_compilation_outputs(
                name = name + ("_no_deps" if archives else ""),
                actions = actions,
                cc_toolchain = cc_toolchain,
                feature_configuration = feature_configuration,
                compilation_outputs = cc_common.create_compilation_outputs(
                    objects = all_objects,
                    pic_objects = all_objects,
                ),
                linking_contexts = linking_contexts,
                disallow_dynamic_library = True,
            )
            library = linking_outputs.library_to_link
            static_lib = library.static_library or library.pic_static_library
        if archives:
            static_lib = _merge_static_libs(
                filename = filename,
                actions = actions,
                cc_toolchain = cc_toolchain,
                feature_configuration = feature_configuration,
                static_libs = ([static_lib] if static_lib else []) + archives,
                is_windows = False,
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
        libraries = depset([static_lib_to_link] + unpacked.dynamic_libraries_to_link),
        user_link_flags = depset(unpacked.user_link_flags),
        additional_inputs = depset(unpacked.additional_inputs),
    )
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([ linker_input ]),
    )
    return linking_context, static_lib

def _link(owner, name, actions, cc_toolchain,
          feature_configuration, linking_contexts,
          def_file=None, is_executable=False, user_link_flags=[],
          is_windows=False, additional_inputs=[]):
    unpacked_linking_context = onedal_cc_common.unpack_linking_contexts(linking_contexts)
    if not is_executable and unpacked_linking_context.objects and unpacked_linking_context.pic_objects:
        fail("Dynamic library {} contains non-PIC object files: {}".format(
            name, unpacked_linking_context.objects))
    all_object_list = unpacked_linking_context.pic_objects + unpacked_linking_context.objects
    libraries_to_link = unpacked_linking_context.libraries_to_link
    object_archive_link_flags = []
    # Files that dependencies name only through `user_link_flags` must be
    # staged for this link as well.
    additional_link_inputs = list(additional_inputs) + unpacked_linking_context.additional_inputs
    if is_windows and all_object_list and not is_executable:
        object_archive = _merge_static_libs(
            filename = name + "_objects.lib",
            actions = actions,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            static_libs = all_object_list,
            is_windows = True,
        )
        object_archive_to_link = cc_common.create_library_to_link(
            actions = actions,
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
            static_library = object_archive,
            pic_static_library = object_archive,
        )
        libraries_to_link = [object_archive_to_link] + libraries_to_link
        # Whole archive, or the dllexport objects stay out of leaf DLLs such as onedal_thread.dll.
        # No `/link` here: dpc_linker_mode emits one already, and a second gives LNK4044.
        object_archive_link_flags.append("/WHOLEARCHIVE:" + object_archive.path)
        additional_link_inputs.append(object_archive)
        all_object_list = []
    all_objects = depset(all_object_list)
    compilation_outputs = cc_common.create_compilation_outputs(
        objects = all_objects,
        pic_objects = all_objects,
    )
    unpacked_user_link_flags = _filter_user_link_flags(
        feature_configuration,
        unpacked_linking_context.user_link_flags
    )
    all_user_link_flags = unpacked_user_link_flags + user_link_flags + object_archive_link_flags
    linker_input = cc_common.create_linker_input(
        owner = owner,
        libraries = depset(libraries_to_link),
        user_link_flags = depset(all_user_link_flags),
    )
    # TODO: pass the compilation outputs per library tag, to link with --start-lib/--end-lib.
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([linker_input]),
    )
    def_file_link_flags = []
    if def_file:
        # `@file` would be read as a response file, exporting everything in the whole archives.
        if not is_windows:
            fail("'{}': def_file is supported on Windows only".format(name))
        def_file_link_flags = ["/DEF:" + def_file.path]
    linking_outputs = cc_common.link(
        name = name,
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        compilation_outputs = compilation_outputs,
        linking_contexts = [linking_context],
        output_type = "executable" if is_executable else "dynamic_library",
        link_deps_statically = True,
        user_link_flags = def_file_link_flags,
        additional_inputs = ([def_file] if def_file else []) + additional_link_inputs,
    )
    return unpacked_linking_context, linking_outputs

def _dynamic(owner, name, actions, cc_toolchain,
             feature_configuration, linking_contexts,
             def_file=None, user_link_flags=[], is_windows=False,
             additional_inputs=[]):
    unpacked_linking_context, linking_outputs = _link(
        owner, name, actions, cc_toolchain,
        feature_configuration, linking_contexts,
        def_file,
        user_link_flags=user_link_flags,
        is_windows=is_windows,
        additional_inputs=additional_inputs,
    )
    library_to_link = linking_outputs.library_to_link
    dynamic_lib = None
    interface_lib = None
    if library_to_link:
        dynamic_lib = library_to_link.resolved_symlink_dynamic_library or library_to_link.dynamic_library
        # On Windows this is the import library the link action declared as
        # its output (see `targets_windows` in cc_toolchain_config_win.bzl).
        interface_lib = library_to_link.resolved_symlink_interface_library or library_to_link.interface_library
    if not dynamic_lib:
        return utils.warn("'{}' dynamic library does not contain any " +
                          "object file".format(name)), struct(
            files = [],
            dynamic_library = None,
            interface_library = None,
        )
    dynamic_files = [dynamic_lib]
    if interface_lib:
        dynamic_files.append(interface_lib)
    dynamic_lib_to_link = cc_common.create_library_to_link(
        actions = actions,
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        dynamic_library = dynamic_lib,
        interface_library = interface_lib,
    )
    linker_input = cc_common.create_linker_input(
        owner = owner,
        libraries = depset([dynamic_lib_to_link] +
                           unpacked_linking_context.dynamic_libraries_to_link),
        user_link_flags = depset(unpacked_linking_context.user_link_flags),
        additional_inputs = depset(unpacked_linking_context.additional_inputs),
    )
    linking_context = cc_common.create_linking_context(
        linker_inputs = depset([ linker_input ]),
    )
    return linking_context, struct(
        files = dynamic_files,
        dynamic_library = dynamic_lib,
        interface_library = interface_lib,
    )

def _executable(owner, name, actions, cc_toolchain,
                feature_configuration, linking_contexts,
                user_link_flags=[]):
    _, linking_outputs = _link(
        owner, name, actions, cc_toolchain,
        feature_configuration, linking_contexts,
        is_executable = True,
        user_link_flags = user_link_flags,
        is_windows = False,
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
