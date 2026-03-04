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

load("@onedal//dev/bazel:utils.bzl",
    "utils",
    "paths",
    "sets",
)
load("@onedal//dev/bazel/cc:common.bzl",
    onedal_cc_common = "common",
)

# TODO: Replace file sufix to ISA
_CPU_SUFFIX_TO_ISA_MAP = {
    "_nrh": "sse2",
    "_neh": "sse42",
    "_hsw": "avx2",
    "_skx": "avx512",
}
_CPU_SUFFIX_TO_ISA_BACK_MAP = {
    "sse2":       "_nrh",
    "sse42":      "_neh",
    "avx2":       "_hsw",
    "avx512":     "_skx",
}
_CPU_SUFFIXES = _CPU_SUFFIX_TO_ISA_MAP.keys()
_CPU_ISA_IDS = _CPU_SUFFIX_TO_ISA_MAP.values()

def _categorize_sources(source_files, cpu_files_supported = True,
                                      fpt_files_supported = True):
    fpt_cpu_files_supported = cpu_files_supported and fpt_files_supported
    normal_files = []
    cpu_files = []
    fpt_files = []
    fpt_cpu_files = []
    special_cpu_files = None
    for file in source_files:
        filename = file.basename
        if fpt_cpu_files_supported and utils.match_substring(filename, ["_fpt_cpu"]):
            fpt_cpu_files.append(file)
        elif fpt_files_supported and utils.match_substring(filename, ["_fpt"]):
            fpt_files.append(file)
        elif cpu_files_supported and utils.match_substring(filename, ["_cpu"]):
            cpu_files.append(file)
        else:
            cpu_suffix = utils.match_substring(filename, _CPU_SUFFIXES)
            if cpu_suffix:
                if not special_cpu_files:
                    special_cpu_files = {v: [] for v in _CPU_ISA_IDS}
                isa = _CPU_SUFFIX_TO_ISA_MAP[cpu_suffix]
                special_cpu_files[isa].append(file)
            else:
                normal_files.append(file)
    return struct(
        normal_files = normal_files,
        cpu_files = cpu_files,
        fpt_files = fpt_files,
        fpt_cpu_files = fpt_cpu_files,
        special_cpu_files = special_cpu_files,
    )

def _normalize_cpu_files(general_cpu_files, special_cpu_files):
    normalized_cpu_files = {}
    general_dict = {}
    for file in general_cpu_files:
        name, _ = paths.split_extension(file.path)
        basename = utils.remove_substring(name, "_cpu")
        general_dict[basename] = file
    for isa, special_files in special_cpu_files.items():
        specific_dict = {}
        suffix = _CPU_SUFFIX_TO_ISA_BACK_MAP[isa]
        for file in special_files:
            name, _ = paths.split_extension(file.path)
            basename = utils.remove_substring(name, suffix)
            specific_dict[basename] = file
        merged_dict = {}
        merged_dict.update(**general_dict)
        merged_dict.update(**specific_dict)
        normalized_cpu_files[isa] = merged_dict.values()
    return normalized_cpu_files

def _normalize_includes(ctx, includes, extra=[]):
    inc_dir = paths.dirname(ctx.build_file_path) + "/"
    gen_dir = ctx.genfiles_dir.path + "/" + inc_dir
    normalized_includes = (utils.add_prefix(inc_dir, includes) +
                           utils.add_prefix(gen_dir, includes))
    normalized_includes = [paths.normalize(x) for x in normalized_includes]
    normalized_includes = utils.unique(normalized_includes + extra)
    return normalized_includes

def _filter_out_includes(includes, filter):
    filter_set = sets.make(filter)
    filtered = []
    for include in includes:
        if not sets.contains(filter_set, include):
            filtered.append(include)
    return filtered

def _patch_includes(ctx, compilation_context, includes=[],
                    system_includes=[], quote_includes=[]):
    system_includes = _normalize_includes(
        ctx,
        system_includes,
        compilation_context.system_includes.to_list(),
    )
    includes = _filter_out_includes(
        _normalize_includes(ctx, includes, compilation_context.includes.to_list()),
        system_includes,
    )
    quote_includes = _filter_out_includes(
        _normalize_includes(ctx, quote_includes, compilation_context.quote_includes.to_list()),
        system_includes,
    )
    return cc_common.create_compilation_context(
        headers = compilation_context.headers,
        system_includes = depset(system_includes),
        includes = depset(includes),
        quote_includes = depset(quote_includes),
        framework_includes = compilation_context.framework_includes,
        defines = compilation_context.defines,
        local_defines = compilation_context.local_defines,
    )

def _compile_wrapper(name, ctx, toolchain, feature_config, **kwargs):
    return cc_common.compile(
        name = name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        disallow_nopic_outputs = True,
        **kwargs,
    )

def _configure_cpu_features(ctx, toolchain, cpus):
    cpu_feature_configs = {}
    for cpu in cpus:
        cpu_feature_configs[cpu] = cc_common.configure_features(
            ctx = ctx,
            cc_toolchain = toolchain,
            requested_features = ctx.features + [ "{}_flags".format(cpu) ],
            unsupported_features = ctx.disabled_features,
        )
    return cpu_feature_configs

def _compile(name, ctx, toolchain, feature_config, compilation_contexts=[],
             cpus=[], fpts=[], cpu_defines={}, fpt_defines={},
             srcs=[], local_defines=[], includes=[], system_includes=[],
             quote_includes=[], **kwargs):

    sources_by_category = _categorize_sources(srcs)
    dep_compilation_context = onedal_cc_common.merge_compilation_contexts(compilation_contexts)
    dep_compilation_context = _patch_includes(
        ctx, dep_compilation_context,
        includes = includes,
        system_includes = system_includes,
        quote_includes = quote_includes,
    )
    dep_compilation_contexts = [ dep_compilation_context ]

    if sources_by_category.cpu_files or sources_by_category.fpt_cpu_files:
        cpu_defines = utils.normalize_dict(cpu_defines, cpus, default=[])
        cpu_feature_configs = _configure_cpu_features(ctx, toolchain, cpus)
    if sources_by_category.fpt_files or sources_by_category.fpt_cpu_files:
        fpt_defines = utils.normalize_dict(fpt_defines, fpts, default=[])

    compilation_contexts = []
    compilation_outputs = []

    # Compile normal files
    compilation_context, compulation_output = _compile_wrapper(
        name, ctx, toolchain, feature_config,
        srcs = sources_by_category.normal_files,
        local_defines = local_defines,
        compilation_contexts = dep_compilation_contexts,
        **kwargs,
    )
    compilation_contexts.append(compilation_context)
    compilation_outputs.append(compulation_output)

    # Compile FPT files
    if sources_by_category.fpt_files:
        for fpt in fpts:
            compilation_context, compulation_output = _compile_wrapper(
                name + "_" + fpt, ctx, toolchain, feature_config,
                srcs = sources_by_category.fpt_files,
                local_defines = local_defines + fpt_defines[fpt],
                compilation_contexts = dep_compilation_contexts,
                **kwargs,
            )
            compilation_contexts.append(compilation_context)
            compilation_outputs.append(compulation_output)

    # Compile CPU files
    if sources_by_category.cpu_files:
        cpu_to_file_dict = {}
        if sources_by_category.special_cpu_files:
            cpu_to_file_dict = _normalize_cpu_files(sources_by_category.cpu_files,
                                                    sources_by_category.special_cpu_files)
        for cpu in cpus:
            compilation_context, compulation_output = _compile_wrapper(
                name + "_" + cpu, ctx, toolchain, cpu_feature_configs[cpu],
                srcs = (cpu_to_file_dict[cpu] if cpu_to_file_dict else
                        sources_by_category.cpu_files),
                local_defines = local_defines + cpu_defines[cpu],
                compilation_contexts = dep_compilation_contexts,
                **kwargs,
            )
            compilation_contexts.append(compilation_context)
            compilation_outputs.append(compulation_output)

    # Compile FPT-CPU files
    if sources_by_category.fpt_cpu_files:
        for cpu in cpus:
            for fpt in fpts:
                compilation_context, compulation_output = _compile_wrapper(
                    name + "_" + fpt + "_" + cpu, ctx, toolchain, cpu_feature_configs[cpu],
                    srcs = sources_by_category.fpt_cpu_files,
                    local_defines = local_defines + cpu_defines[cpu] + fpt_defines[fpt],
                    compilation_contexts = dep_compilation_contexts,
                    **kwargs,
                )
                compilation_contexts.append(compilation_context)
                compilation_outputs.append(compulation_output)

    compilation_context = onedal_cc_common.merge_compilation_contexts(
        compilation_contexts = compilation_contexts,
    )
    compilation_output = cc_common.merge_compilation_outputs(
        compilation_outputs = compilation_outputs,
    )
    return compilation_context, compilation_output

compile = struct(
    compile = _compile,
)
