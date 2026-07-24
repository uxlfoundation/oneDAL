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

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@onedal//dev/bazel/config:config.bzl",
    "ConfigFlagInfo",
    "CpuInfo",
    "VersionInfo",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@onedal//dev/bazel/cc:common.bzl",
    onedal_cc_common = "common",
)
load("@onedal//dev/bazel/cc:compile.bzl",
    onedal_cc_compile = "compile",
)
load("@onedal//dev/bazel/cc:link.bzl",
    onedal_cc_link = "link",
)

ModuleInfo = onedal_cc_common.ModuleInfo

def _collect_runfiles(deps):
    runfiles = []
    for dep in deps:
        if DefaultInfo in dep:
            runfiles.append(dep[DefaultInfo].default_runfiles)
            runfiles.append(dep[DefaultInfo].data_runfiles)
    return runfiles

def _copy_windows_runtime_files(ctx, files, is_windows):
    if not is_windows or not ctx.attr._is_test:
        return []
    outputs = []
    for src in files:
        if src.extension.lower() != "dll":
            continue
        out = ctx.actions.declare_file(src.basename)
        ctx.actions.run(
            executable = "cmd.exe",
            inputs = [src],
            outputs = [out],
            arguments = [
                "/d",
                "/c",
                'copy /Y "{}" "{}"'.format(
                    src.path.replace("/", "\\"),
                    out.path.replace("/", "\\"),
                ),
            ],
            use_default_shell_env = True,
        )
        outputs.append(out)
    return outputs

def _init_cc_rule(ctx, features=[], disable_features=[]):
    toolchain = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]
    cc_toolchain = toolchain.cc
    feature_config = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features + features,
        unsupported_features = ctx.disabled_features + disable_features,
    )
    return cc_toolchain, feature_config


def _coverage_options(ctx, cc_toolchain, feature_config):
    """Return Make-compatible coverage options for oneDAL-owned actions."""
    if not ctx.attr._code_coverage[ConfigFlagInfo].flag:
        return struct(compile_flags = [], local_defines = [], link_flags = [])

    is_linux = ctx.target_platform_has_constraint(
        ctx.attr._linux_constraint[platform_common.ConstraintValueInfo],
    )
    compiler_id = cc_toolchain.compiler.split("-")[0]
    if not is_linux or compiler_id != "icx":
        fail("--code_coverage=true is supported only on Linux with the Intel " +
             "icx/icpx toolchain (selected compiler: '{}')".format(
                 cc_toolchain.compiler,
             ))

    is_dpc = cc_common.is_enabled(
        feature_configuration = feature_config,
        feature_name = "dpc++",
    )
    # Make adds -coverage to ICX compile and link actions. DPC++ compile
    # actions are unchanged, while DPC++ link actions receive -Xscoverage.
    return struct(
        compile_flags = [] if is_dpc else ["-coverage"],
        local_defines = ["GCOV_BUILD"],
        link_flags = ["-Xscoverage"] if is_dpc else ["-coverage"],
    )

def _cc_module_impl(ctx):
    toolchain, feature_config = _init_cc_rule(ctx)
    coverage = _coverage_options(ctx, toolchain, feature_config)
    dep_compilation_contexts = onedal_cc_common.collect_compilation_contexts(ctx.attr.deps)
    is_windows = ctx.target_platform_has_constraint(
        ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
    )
    compilation_context, compilation_outputs = onedal_cc_compile.compile(
        name = ctx.label.name,
        ctx = ctx,
        toolchain = toolchain,
        feature_config = feature_config,
        compilation_contexts = dep_compilation_contexts,
        fpts = ctx.attr._fpts,
        cpus = ctx.attr._cpus[CpuInfo].enabled,
        cpu_defines = ctx.attr.cpu_defines,
        fpt_defines = ctx.attr.fpt_defines,
        srcs = ctx.files.srcs,
        public_hdrs = ctx.files.hdrs,
        private_hdrs = ctx.files.private_hdrs,
        defines = ctx.attr.defines,
        local_defines = ctx.attr.local_defines + coverage.local_defines,
        user_compile_flags = ctx.attr.copts + coverage.compile_flags,
        includes = ctx.attr.includes,
        system_includes = ctx.attr.system_includes,
        quote_includes = ctx.attr.quote_includes,
        disallow_nopic_outputs = not is_windows,
    )
    if compilation_outputs.objects and compilation_outputs.pic_objects:
        fail("Non-PIC object files found, oneDAL assumes " +
             "all object files are compiled as PIC")
    tagged_linking_contexts = onedal_cc_common.collect_tagged_linking_contexts(ctx.attr.deps)
    if ctx.attr.override_deps_lib_tag:
        tagged_linking_contexts = onedal_cc_common. \
            override_tags(tagged_linking_contexts, ctx.attr.lib_tag)
    if compilation_outputs.objects or compilation_outputs.pic_objects:
        linking_context, linking_out = cc_common.create_linking_context_from_compilation_outputs(
            name = ctx.label.name,
            actions = ctx.actions,
            cc_toolchain = toolchain,
            feature_configuration = feature_config,
            compilation_outputs = compilation_outputs,
        )
        tagged_linking_contexts.append(onedal_cc_common.create_tagged_linking_context(
            tag = ctx.attr.lib_tag,
            linking_context = linking_context,
        ))
    module_info = ModuleInfo(
        compilation_context = compilation_context,
        tagged_linking_contexts = tagged_linking_contexts,
    )
    default_info = DefaultInfo(
        runfiles = ctx.runfiles().merge_all(_collect_runfiles(ctx.attr.deps)),
    )
    return [module_info, default_info]

_cc_module = rule(
    implementation = _cc_module_impl,
    attrs = {
        "lib_tag": attr.string(),
        "override_deps_lib_tag": attr.bool(default=False),
        "srcs": attr.label_list(allow_files=True),
        "hdrs": attr.label_list(allow_files=True),
        "private_hdrs": attr.label_list(allow_files=True),
        "deps": attr.label_list(),
        "copts": attr.string_list(),
        "defines": attr.string_list(),
        "local_defines": attr.string_list(),
        "cpu_defines": attr.string_list_dict(),
        "fpt_defines": attr.string_list_dict(),
        "includes": attr.string_list(),
        "quote_includes": attr.string_list(),
        "system_includes": attr.string_list(),
        "_cpus": attr.label(
            default = "@config//:cpu",
        ),
        "_fpts": attr.string_list(default = ["f32", "f64"]),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
        "_linux_constraint": attr.label(
            default = "@platforms//os:linux",
        ),
        "_code_coverage": attr.label(
            default = "@config//:code_coverage",
            providers = [ConfigFlagInfo],
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)

def cc_module(name, hdrs=[], deps=[], **kwargs):
    # Workaround restriction on possible extensions for cc_common.compile:
    # > The list of possible extensions for 'public_hdrs' is:
    #   .h,.hh,.hpp,.ipp,.hxx,.h++,.inc,.inl,.tlh,.tli,.H,.tcc
    if hdrs:
        cc_library(
            name = "__{}_headers__".format(name),
            hdrs = hdrs,
        )
    _cc_module(
        name = name,
        deps = deps + ([
            ":__{}_headers__".format(name),
        ] if hdrs else []),
        **kwargs,
    )


def _cc_static_lib_impl(ctx):
    toolchain, feature_config = _init_cc_rule(ctx)
    compilation_context = onedal_cc_common.collect_and_merge_compilation_contexts(ctx.attr.deps)
    linking_contexts = onedal_cc_common.collect_and_filter_linking_contexts(
        ctx.attr.deps, ctx.attr.lib_tags)
    is_windows = ctx.target_platform_has_constraint(
        ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
    )
    linking_context, static_lib = onedal_cc_link.static(
        owner = ctx.label,
        name = ctx.attr.lib_name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        linking_contexts = linking_contexts,
        is_windows = is_windows,
    )
    default_info = DefaultInfo(
        files = depset([ static_lib ]),
    )
    cc_info = CcInfo(
        compilation_context = compilation_context,
        linking_context = linking_context,
    )
    return [default_info, cc_info]

cc_static_lib = rule(
    implementation = _cc_static_lib_impl,
    attrs = {
        "lib_name": attr.string(),
        "lib_tags": attr.string_list(),
        "deps": attr.label_list(mandatory=True),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)


def _copy_dynamic_release_file(ctx, src, out_name, is_windows = False, extra_inputs = []):
    out = ctx.actions.declare_file(out_name)
    if is_windows:
        ctx.actions.run(
            executable = "cmd.exe",
            inputs = [src] + extra_inputs,
            outputs = [out],
            arguments = [
                "/d",
                "/c",
                'copy /Y "{}" "{}"'.format(
                    src.path.replace("/", "\\"),
                    out.path.replace("/", "\\"),
                ),
            ],
            use_default_shell_env = True,
        )
    else:
        ctx.actions.run(
            executable = "cp",
            inputs = [src] + extra_inputs,
            outputs = [out],
            arguments = [src.path, out.path],
            use_default_shell_env = True,
        )
    return out


def _cc_dynamic_lib_impl(ctx):
    toolchain, feature_config = _init_cc_rule(ctx, features=[
        # Keep produced shared libraries free of dynamic Bazel deps on Linux.
        # Release-dynamic tests provide standalone oneDAL libs through
        # DALROOT/LD_LIBRARY_PATH while Bazel-provided runtimes such as TBB
        # live in test runfiles; DT_NEEDED entries on those deps are not
        # resolved through the executable RUNPATH.
        "do_not_link_dynamic_dependencies",
    ])
    coverage = _coverage_options(ctx, toolchain, feature_config)
    compilation_context = onedal_cc_common.collect_and_merge_compilation_contexts(ctx.attr.deps)
    linking_contexts = onedal_cc_common.collect_and_filter_linking_contexts(
        ctx.attr.deps, ctx.attr.lib_tags)

    is_windows = ctx.target_platform_has_constraint(
        ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
    )
    vi = ctx.attr._version_info[VersionInfo]
    link_name = "{}.{}".format(ctx.attr.lib_name, vi.binary_major) if is_windows else ctx.attr.lib_name
    linux_soname_flags = [] if is_windows else [
        "-Wl,-soname,lib{}.so.{}".format(ctx.attr.lib_name, vi.binary_major),
    ]
    linux_linker_script_flags = [] if is_windows else [
        "-Wl,--version-script,{}".format(script.path)
        for script in ctx.files.linker_scripts
    ]
    linking_context, dynamic_outputs = onedal_cc_link.dynamic(
        owner = ctx.label,
        name = link_name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        linking_contexts = linking_contexts,
        def_file = ctx.file.def_file,
        user_link_flags = linux_soname_flags + linux_linker_script_flags + (["-Wl,--exclude-libs=ALL"] if not is_windows else []) + ctx.attr.linkopts + coverage.link_flags,
        is_windows = is_windows,
        additional_inputs = ctx.files.linker_scripts,
        is_dpc = ctx.attr.lib_name.endswith("_dpc") or ctx.label.name.endswith("_dpc"),
    )
    default_files = dynamic_outputs.files
    if is_windows:
        default_files = []
        if dynamic_outputs.dynamic_library:
            dynamic_release_name = "{}.{}.dll".format(ctx.attr.lib_name, vi.binary_major)
            if dynamic_outputs.dynamic_library.basename == dynamic_release_name:
                default_files.append(dynamic_outputs.dynamic_library)
            else:
                default_files.append(_copy_dynamic_release_file(
                    ctx,
                    dynamic_outputs.dynamic_library,
                    dynamic_release_name,
                    is_windows = is_windows,
                    extra_inputs = [dynamic_outputs.interface_library] if dynamic_outputs.interface_library else [],
                ))
        if dynamic_outputs.interface_library:
            default_files.append(_copy_dynamic_release_file(
                ctx,
                dynamic_outputs.interface_library,
                "{}_dll.lib".format(ctx.attr.lib_name),
                is_windows = is_windows,
            ))
    default_info = DefaultInfo(
        files = depset(default_files),
    )
    cc_info = CcInfo(
        compilation_context = compilation_context,
        linking_context = linking_context,
    )
    return [default_info, cc_info]

cc_dynamic_lib = rule(
    implementation = _cc_dynamic_lib_impl,
    attrs = {
        "lib_name": attr.string(),
        "lib_tags": attr.string_list(),
        "deps": attr.label_list(mandatory=True),
        "def_file": attr.label(allow_single_file=True),
        "linkopts": attr.string_list(
            default = [],
            doc = "Additional linker flags (e.g. --exclude-libs for MKL symbol hiding).",
        ),
        "linker_scripts": attr.label_list(allow_files=True),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
        "_linux_constraint": attr.label(
            default = "@platforms//os:linux",
        ),
        "_code_coverage": attr.label(
            default = "@config//:code_coverage",
            providers = [ConfigFlagInfo],
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)


def _cc_exec_impl(ctx):
    if not ctx.attr.deps:
        return
    toolchain, feature_config = _init_cc_rule(ctx)
    coverage = _coverage_options(ctx, toolchain, feature_config)
    is_windows = ctx.target_platform_has_constraint(
        ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
    )
    tagged_linking_contexts = onedal_cc_common.collect_tagged_linking_contexts(ctx.attr.deps)
    linking_contexts = onedal_cc_common.filter_tagged_linking_contexts(
        tagged_linking_contexts, ctx.attr.lib_tags)
    executable = onedal_cc_link.executable(
        owner = ctx.label,
        name = ctx.label.name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        linking_contexts = linking_contexts,
        user_link_flags = ctx.attr.user_link_flags + coverage.link_flags,
    )
    runtime_files = _copy_windows_runtime_files(ctx, ctx.files.data, is_windows)
    default_info = DefaultInfo(
        files = depset([executable] + runtime_files),
        runfiles = ctx.runfiles(files = ctx.files.data + runtime_files).merge_all(
            _collect_runfiles(ctx.attr.deps + ctx.attr.data),
        ),
        executable = executable,
    )
    return [default_info]

cc_test = rule(
    implementation = _cc_exec_impl,
    attrs = {
        "lib_tags": attr.string_list(),
        "deps": attr.label_list(),
        "data": attr.label_list(allow_files=True),
        "user_link_flags": attr.string_list(),
        "_is_test": attr.bool(default=True),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
        "_linux_constraint": attr.label(
            default = "@platforms//os:linux",
        ),
        "_code_coverage": attr.label(
            default = "@config//:code_coverage",
            providers = [ConfigFlagInfo],
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
    test = True,
)

cc_executable = rule(
    implementation = _cc_exec_impl,
    attrs = {
        "lib_tags": attr.string_list(),
        "deps": attr.label_list(),
        "data": attr.label_list(allow_files=True),
        "user_link_flags": attr.string_list(),
        "_is_test": attr.bool(default=False),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
        "_linux_constraint": attr.label(
            default = "@platforms//os:linux",
        ),
        "_code_coverage": attr.label(
            default = "@config//:code_coverage",
            providers = [ConfigFlagInfo],
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
    executable = True,
)
