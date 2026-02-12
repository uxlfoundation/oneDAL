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
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
_ModuleInfo = provider(
    fields = [
        "compilation_context",
        "tagged_linking_contexts",
    ]
)

_EnvInfo = provider(
    fields = [
        "var",
        "value",
    ]
)

def _collect_env(deps):
    env_dict = {}
    for dep in deps:
        if _EnvInfo in dep:
            env_info = dep[_EnvInfo]
            env_dict[env_info.var] = env_info.value
    return env_dict;

def _collect_compilation_contexts(deps):
    dep_compilation_contexts = []
    for dep in deps:
        for Info in [CcInfo, _ModuleInfo]:
            if Info in dep:
                dep_compilation_contexts.append(dep[Info].compilation_context)
    return dep_compilation_contexts

def _merge_compilation_contexts(compilation_contexts):
    cc_infos = [CcInfo(compilation_context=x) for x in compilation_contexts]
    return cc_common.merge_cc_infos(
        direct_cc_infos = cc_infos
    ).compilation_context

def _collect_and_merge_compilation_contexts(deps):
    compilation_contexts = _collect_compilation_contexts(deps)
    compilation_contexts = _merge_compilation_contexts(compilation_contexts)
    return compilation_contexts

def _create_tagged_linking_context(tag, linking_context):
    return struct(
        tag = tag,
        linking_context = linking_context,
    )

def _collect_tagged_linking_contexts(deps):
    # TODO: Merge linking contexts with the same tag to minimize amount
    #       of linking contexts need to be collected by the modules
    dep_tagged_linking_contexts = []
    for dep in deps:
        if _ModuleInfo in dep:
            dep_tagged_linking_contexts += dep[_ModuleInfo].tagged_linking_contexts
        if CcInfo in dep:
            linking_context = dep[CcInfo].linking_context
            dep_tagged_linking_contexts.append(_create_tagged_linking_context(
                tag = None,
                linking_context = linking_context,
            ))
    return dep_tagged_linking_contexts

def _filter_tagged_linking_contexts(tagged_linking_contexts, tags):
    linking_contexts = []
    tag_set = sets.make(tags)
    for tagged_linking_context in tagged_linking_contexts:
        tag = tagged_linking_context.tag
        linking_context = tagged_linking_context.linking_context
        if (not tag) or (not tags) or sets.contains(tag_set, tag):
            linking_contexts.append(linking_context)
    return linking_contexts

def _collect_and_filter_linking_contexts(deps, tags):
    tagged_linking_contexts = _collect_tagged_linking_contexts(deps)
    linking_contexts = _filter_tagged_linking_contexts(tagged_linking_contexts, tags)
    return linking_contexts

def _unpack_linking_contexts(linking_contexts):
    link_flags = []
    objects = []
    pic_objects = []
    dynamic_libs = []
    static_libs = []
    libs_to_link = []
    dynamic_libs_to_link = []
    static_libs_to_link = []
    # IMPORTANT: We need to preserve order of objects when
    #            iterate over linking contexts
    for linking_context in linking_contexts:
        for linker_input in linking_context.linker_inputs.to_list():
            for lib_to_link in linker_input.libraries:
                if lib_to_link.objects:
                    objects += lib_to_link.objects
                elif lib_to_link.pic_objects:
                    pic_objects += lib_to_link.pic_objects
                elif lib_to_link.dynamic_library:
                    libs_to_link.append(lib_to_link)
                    dynamic_libs_to_link.append(lib_to_link)
                    dynamic_libs.append(lib_to_link.dynamic_library)
                elif lib_to_link.static_library or lib_to_link.pic_static_library:
                    libs_to_link.append(lib_to_link)
                    static_libs_to_link.append(lib_to_link)
                    static_libs.append(lib_to_link.static_library or
                                       lib_to_link.pic_static_library)
            link_flags += linker_input.user_link_flags
    return struct(
        pic_objects = depset(pic_objects).to_list(),
        objects = depset(objects).to_list(),
        dynamic_libraries = depset(dynamic_libs).to_list(),
        dynamic_libraries_to_link = dynamic_libs_to_link,
        static_libraries = depset(static_libs).to_list(),
        static_libraries_to_link = static_libs_to_link,
        libraries_to_link = libs_to_link,
        user_link_flags = utils.unique(link_flags),
    )

def _override_tags(tagged_linking_contexts, tag):
    overridden = []
    for tagged_linking_context in tagged_linking_contexts:
        overridden.append(_create_tagged_linking_context(
            tag = tag,
            linking_context = tagged_linking_context.linking_context,
        ))
    return overridden

common = struct(
    ModuleInfo = _ModuleInfo,
    EnvInfo = _EnvInfo,
    collect_env = _collect_env,
    collect_compilation_contexts = _collect_compilation_contexts,
    merge_compilation_contexts = _merge_compilation_contexts,
    collect_and_merge_compilation_contexts = _collect_and_merge_compilation_contexts,
    create_tagged_linking_context = _create_tagged_linking_context,
    collect_tagged_linking_contexts = _collect_tagged_linking_contexts,
    filter_tagged_linking_contexts = _filter_tagged_linking_contexts,
    collect_and_filter_linking_contexts = _collect_and_filter_linking_contexts,
    unpack_linking_contexts = _unpack_linking_contexts,
    override_tags = _override_tags,
)
