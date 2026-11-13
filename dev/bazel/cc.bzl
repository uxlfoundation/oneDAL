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
    "paths",
)

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@onedal//dev/bazel/config:config.bzl",
    "CPU_ATTRS",
    "ConfigFlagInfo",
    "TARGET_ATTRS",
    "VERSION_ATTRS",
    "get_cpu_info",
    "target",
    "validate_build_parameters_lib",
    "get_version_info",
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

_CODE_COVERAGE_ATTRS = {
    "_code_coverage": attr.label(default = "@config//:code_coverage", providers = [ConfigFlagInfo]),
}

_PARAMETERS_LIB_ATTRS = {
    "_build_parameters_lib": attr.label(default = "@config//:build_parameters_lib", providers = [ConfigFlagInfo]),
}

ModuleInfo = onedal_cc_common.ModuleInfo

def _collect_runfiles(deps):
    runfiles = []
    for dep in deps:
        if DefaultInfo in dep:
            runfiles.append(dep[DefaultInfo].default_runfiles)
            runfiles.append(dep[DefaultInfo].data_runfiles)
    return runfiles

def _copy_windows_runtime_files(ctx, files):
    """Put the DLLs a test loads next to its executable, where the loader looks.

    `symlink` copies the file unless `--windows_enable_symlinks` is set.
    """
    if not target(ctx).is_windows or not ctx.attr._is_test:
        return []
    outputs = []
    for src in files:
        if src.extension.lower() != "dll":
            continue
        out = ctx.actions.declare_file(src.basename)
        ctx.actions.symlink(output = out, target_file = src)
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
    _check_msvc_runtime(target(ctx), feature_config)
    return cc_toolchain, feature_config

def _missing_dpc_compiler_inputs(ctx, feature_config):
    """Returns a failing action's output for the compiles of a DPC++ target the toolchain cannot build.

    The error waits for the build, so `--nobuild` still analyses the DPC++ targets without icpx.
    The host driver preprocesses an `#error`, which every supported compiler accepts with `-E`.
    """
    if not (cc_common.is_enabled(feature_configuration = feature_config, feature_name = "dpc++") and
            not cc_common.is_enabled(feature_configuration = feature_config, feature_name = "has_dpc_compiler")):
        return []
    message = ("{} needs the DPC++ compiler (icpx on Linux, icx on Windows; clang-cl has none), " +
               "but the C++ toolchain was configured without it. Put it in PATH and do not select " +
               "another compiler through CC/CXX or ONEDAL_WIN_COMPILER, or pass --release_dpc=false " +
               "to skip the DPC++ targets.").format(ctx.label)
    src = ctx.actions.declare_file(ctx.label.name + "_needs_dpc_compiler.c")
    ctx.actions.write(src, "#error \"{}\"\n".format(message.replace("\"", "'")))
    out = ctx.actions.declare_file(ctx.label.name + "_needs_dpc_compiler.i")
    ctx.actions.run(
        executable = cc_common.get_tool_for_action(feature_configuration = feature_config, action_name = "c-compile"),
        arguments = ["-E", src.path],
        inputs = [src],
        outputs = [out],
        env = cc_common.get_environment_variables(
            feature_configuration = feature_config,
            action_name = "c-compile",
            variables = cc_common.empty_variables(),
        ),
        mnemonic = "DpcCompilerCheck",
        progress_message = "Checking for the DPC++ compiler for %{label}",
    )
    return [out]

def _coverage_options(ctx, cc_toolchain, feature_config):
    """Return Make-compatible coverage driver options for oneDAL actions."""
    if not ctx.attr._code_coverage[ConfigFlagInfo].flag:
        return struct(compile_flags = [], link_flags = [])

    compiler_id = cc_toolchain.compiler.split("-")[0]
    if not target(ctx).is_linux:
        fail("--code_coverage=true requires Linux (detected target OS is not Linux)")
    # `CC=icpx` is detected as `icpx`; it is the same Intel driver as `icx`.
    if compiler_id not in ("icx", "icpx"):
        fail("--code_coverage=true requires host compiler ID 'icx' or 'icpx' " +
             "(detected compiler ID: '{}')".format(compiler_id))

    is_dpc = cc_common.is_enabled(
        feature_configuration = feature_config,
        feature_name = "dpc++",
    )
    # Make adds -coverage to every ICX compilation and to dynamic links.
    # DPC++ compilations are unchanged; DPC++ dynamic links use -Xscoverage.
    return struct(
        compile_flags = [] if is_dpc else ["-coverage"],
        link_flags = ["-Xscoverage"] if is_dpc else ["-coverage"],
    )

def _without_coverage_link_flags(linking_context, coverage_link_flags):
    """Remove action-local coverage flags while preserving linker metadata."""
    linker_inputs = []
    for linker_input in linking_context.linker_inputs.to_list():
        linker_inputs.append(cc_common.create_linker_input(
            owner = linker_input.owner,
            libraries = depset(linker_input.libraries),
            user_link_flags = depset([
                flag
                for flag in linker_input.user_link_flags
                if flag not in coverage_link_flags
            ]),
            additional_inputs = depset(linker_input.additional_inputs),
            linkstamps = depset(linker_input.linkstamps),
        ))
    return cc_common.create_linking_context(
        linker_inputs = depset(linker_inputs),
    )

def _check_msvc_runtime(target_info, feature_config):
    """Fail unless the toolchain's CRT choice follows `--msvc_runtime`.

    They diverge on rules_cc's MSVC `cl` config or with `--features=msvc_runtime_debug`.
    Checked in every rule, so tests that link a prebuilt release tree are covered too.
    """
    if not target_info.is_windows:
        return
    feature_enabled = cc_common.is_enabled(
        feature_configuration = feature_config,
        feature_name = "msvc_runtime_debug",
    )
    flag_debug = target_info.msvc_runtime_debug
    if feature_enabled != flag_debug:
        fail(
            ("--msvc_runtime={} but the C++ toolchain {} the msvc_runtime_debug " +
             "feature. The debug MSVC runtime is implemented for the icx and " +
             "clang-cl toolchains only, and --features must not contradict " +
             "--msvc_runtime; select the runtime with --msvc_runtime (--config=mdd).")
            .format("debug" if flag_debug else "release",
                    "enables" if feature_enabled else "does not enable"),
        )

def _msvc_runtime_suffix(ctx):
    """The `d` the debug MSVC CRT appends to library names, as Make's `$d`."""
    return "d" if target(ctx).msvc_runtime_debug else ""

def _cc_module_impl(ctx):
    toolchain, feature_config = _init_cc_rule(ctx)
    coverage = _coverage_options(ctx, toolchain, feature_config)
    coverage_defines = ["GCOV_BUILD"] if (
        ctx.attr.define_gcov_build and ctx.attr._code_coverage[ConfigFlagInfo].flag
    ) else []
    dep_compilation_contexts = onedal_cc_common.collect_compilation_contexts(ctx.attr.deps)
    is_windows = target(ctx).is_windows
    compilation_context, compilation_outputs = onedal_cc_compile.compile(
        name = ctx.label.name,
        ctx = ctx,
        toolchain = toolchain,
        feature_config = feature_config,
        compilation_contexts = dep_compilation_contexts,
        fpts = ctx.attr._fpts,
        cpus = get_cpu_info(ctx).enabled,
        cpu_defines = ctx.attr.cpu_defines,
        fpt_defines = ctx.attr.fpt_defines,
        srcs = ctx.files.srcs,
        public_hdrs = ctx.files.hdrs,
        private_hdrs = ctx.files.private_hdrs,
        defines = ctx.attr.defines,
        local_defines = ctx.attr.local_defines + coverage_defines,
        user_compile_flags = ctx.attr.copts + coverage.compile_flags,
        includes = ctx.attr.includes,
        system_includes = ctx.attr.system_includes,
        quote_includes = ctx.attr.quote_includes,
        disallow_nopic_outputs = not is_windows,
        short_paths = is_windows,
        drop_mkl_ilp64 = ctx.attr.drop_mkl_ilp64,
        additional_inputs = _missing_dpc_compiler_inputs(ctx, feature_config),
    )
    if compilation_outputs.objects and compilation_outputs.pic_objects:
        fail("Non-PIC object files found, oneDAL assumes " +
             "all object files are compiled as PIC")
    tagged_linking_contexts = onedal_cc_common.collect_tagged_linking_contexts(ctx.attr.deps)
    if ctx.attr.override_deps_lib_tag:
        tagged_linking_contexts = onedal_cc_common. \
            override_tags(tagged_linking_contexts, ctx.attr.lib_tag)
    if compilation_outputs.objects or compilation_outputs.pic_objects:
        # Dependents link the objects, so the per-module shared library is never needed.
        linking_context, linking_out = cc_common.create_linking_context_from_compilation_outputs(
            name = ctx.label.name,
            actions = ctx.actions,
            cc_toolchain = toolchain,
            feature_configuration = feature_config,
            compilation_outputs = compilation_outputs,
            user_link_flags = coverage.link_flags,
            disallow_dynamic_library = True,
        )
        # The module's coverage link flags would otherwise leak into every dependent link,
        # while executables and tests add their own.
        linking_context = _without_coverage_link_flags(linking_context, coverage.link_flags)
        tagged_linking_contexts.append(onedal_cc_common.create_tagged_linking_context(
            tag = ctx.attr.lib_tag,
            linking_context = linking_context,
        ))
    if not ctx.attr.export_hdrs:
        # Dependents compile against the deps' headers only. The module's own
        # headers, private ones included, stay out of their action inputs.
        compilation_context = onedal_cc_common.merge_compilation_contexts(
            dep_compilation_contexts,
        )
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
        "export_hdrs": attr.bool(default=True),
        # Set for DPC++ library modules (see dal.bzl `_dal_module`): their
        # objects are compiled without MKL_ILP64, as Make's DPC objects are.
        "drop_mkl_ilp64": attr.bool(default=False),
        "deps": attr.label_list(),
        "copts": attr.string_list(),
        "defines": attr.string_list(),
        "local_defines": attr.string_list(),
        "define_gcov_build": attr.bool(
            default = False,
            doc = "Define GCOV_BUILD for Make CORE.objs_a/y-equivalent sources.",
        ),
        "cpu_defines": attr.string_list_dict(),
        "fpt_defines": attr.string_list_dict(),
        "includes": attr.string_list(),
        "quote_includes": attr.string_list(),
        "system_includes": attr.string_list(),
        "_fpts": attr.string_list(default = ["f32", "f64"]),
    } | TARGET_ATTRS | CPU_ATTRS | _CODE_COVERAGE_ATTRS,
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)

def cc_module(name, hdrs=[], deps=[], target_compatible_with=[], **kwargs):
    # With --release_dpc=false, the default when the toolchain has no DPC++ compiler, skip
    # DPC++ modules and everything that depends on them.
    if "dpc++" in kwargs.get("features", []):
        target_compatible_with = target_compatible_with + select({
            "@config//:release_dpc_disabled": ["@config//:dpc_needs_icpx_in_PATH"],
            "//conditions:default": [],
        })
    # cc_common.compile restricts the extensions of `public_hdrs`; cc_library does not.
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
        target_compatible_with = target_compatible_with,
        **kwargs,
    )


def _validate_build_parameters_lib(ctx):
    validate_build_parameters_lib(ctx.attr._build_parameters_lib[ConfigFlagInfo].flag, target(ctx).is_windows)

def _cc_static_lib_impl(ctx):
    is_windows = target(ctx).is_windows
    _validate_build_parameters_lib(ctx)
    toolchain, feature_config = _init_cc_rule(ctx)
    compilation_context = onedal_cc_common.collect_and_merge_compilation_contexts(ctx.attr.deps)
    linking_contexts = onedal_cc_common.collect_and_filter_linking_contexts(
        ctx.attr.deps, ctx.attr.lib_tags)
    rt_suffix = _msvc_runtime_suffix(ctx)
    linking_context, static_lib = onedal_cc_link.static(
        owner = ctx.label,
        name = ctx.attr.lib_name + rt_suffix,
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

_STATIC_EXPORT_FREE_SETTING = str(Label("@config//:static_export_free"))

def _static_export_free_transition_impl(settings, attr):
    # Export-free static archives: no __DAAL_IMPLEMENTATION or ONEDAL export defines, as Make's static objects.
    # Applied only when static_export_free is True (Windows cc_static_lib); elsewhere unchanged.
    if attr.static_export_free:
        return {_STATIC_EXPORT_FREE_SETTING: True}
    return {_STATIC_EXPORT_FREE_SETTING: settings[_STATIC_EXPORT_FREE_SETTING]}

_static_export_free_transition = transition(
    implementation = _static_export_free_transition_impl,
    inputs = [_STATIC_EXPORT_FREE_SETTING],
    outputs = [_STATIC_EXPORT_FREE_SETTING],
)

_cc_static_lib = rule(
    implementation = _cc_static_lib_impl,
    attrs = {
        "lib_name": attr.string(),
        "lib_tags": attr.string_list(),
        # Transitioned: the deps (and their own deps) are compiled export-free
        # when `static_export_free` is set.
        "deps": attr.label_list(mandatory=True, cfg=_static_export_free_transition),
        "static_export_free": attr.bool(default=False),
    } | TARGET_ATTRS | _PARAMETERS_LIB_ATTRS,
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)


def cc_static_lib(name, **kwargs):
    # Only Windows archives are export-free; see the transition above.
    _cc_static_lib(
        name = name,
        static_export_free = select({
            "@platforms//os:windows": True,
            "//conditions:default": False,
        }),
        **kwargs
    )

def _compile_windows_resource(ctx, cc_toolchain, rc_file, rc_headers, defines, out_name):
    """Compile a Windows resource script as Make's `RC.COMPILE` does.

    clang uses `llvm-rc`, icx and MSVC `rc.exe`. Each header's directory is an include path.
    """
    out = ctx.actions.declare_file(out_name)
    rc_tool = "llvm-rc" if cc_toolchain.compiler.split("-")[0] == "clang" else "rc.exe"
    include_dirs = []
    for header in rc_headers:
        if header.dirname not in include_dirs:
            include_dirs.append(header.dirname)
    ctx.actions.run(
        executable = "cmd.exe",
        inputs = [rc_file] + rc_headers,
        outputs = [out],
        arguments = ["/d", "/c", rc_tool] +
                    ["-D" + define for define in defines] +
                    ["-I" + d.replace("/", "\\") for d in include_dirs] +
                    ["-fo" + out.path.replace("/", "\\"), rc_file.path.replace("/", "\\")],
        use_default_shell_env = True,
        mnemonic = "WindowsResource",
        progress_message = "Compiling resource script %s" % rc_file.short_path,
    )
    return out

def _cc_dynamic_lib_impl(ctx):
    is_windows = target(ctx).is_windows
    _validate_build_parameters_lib(ctx)
    # No do_not_link_dynamic_dependencies: released libraries need their DT_NEEDED entries
    # (TBB, libstdc++), and a DLL needs the import libraries of the DLLs it uses.
    toolchain, feature_config = _init_cc_rule(ctx)
    coverage = _coverage_options(ctx, toolchain, feature_config)
    compilation_context = onedal_cc_common.collect_and_merge_compilation_contexts(ctx.attr.deps)
    linking_contexts = onedal_cc_common.collect_and_filter_linking_contexts(
        ctx.attr.deps, ctx.attr.lib_tags)

    vi = get_version_info(ctx)
    # The CRT suffix goes before the ABI version, as in Make: `onedal_cored.4.dll`.
    rt_name = ctx.attr.lib_name + _msvc_runtime_suffix(ctx)
    link_name = "{}.{}".format(rt_name, vi.binary_major) if is_windows else rt_name
    linux_soname_flags = [] if is_windows else [
        "-Wl,-soname,lib{}.so.{}".format(ctx.attr.lib_name, vi.binary_major),
    ]
    linux_linker_script_flags = [] if is_windows else [
        "-Wl,--version-script,{}".format(script.path)
        for script in ctx.files.linker_scripts
    ]
    exclude_libs_flags = [] if is_windows else ["-Wl,--exclude-libs=ALL"]
    # Make links the version resource into every Windows DLL (dll.res, makefile:586, 870, 874).
    # Only this DLL links it, so it is kept out of the returned linking context.
    res_files = []
    if is_windows and ctx.file.rc_file:
        res_files.append(_compile_windows_resource(
            ctx,
            toolchain,
            ctx.file.rc_file,
            ctx.files.rc_headers,
            ctx.attr.rc_defines,
            rt_name + ".res",
        ))
    linking_context, dynamic_outputs = onedal_cc_link.dynamic(
        owner = ctx.label,
        name = link_name,
        actions = ctx.actions,
        cc_toolchain = toolchain,
        feature_configuration = feature_config,
        linking_contexts = linking_contexts,
        def_file = ctx.file.def_file,
        user_link_flags = linux_soname_flags + linux_linker_script_flags + exclude_libs_flags +
                          ctx.attr.linkopts + coverage.link_flags + [res.path for res in res_files],
        is_windows = is_windows,
        additional_inputs = ctx.files.linker_scripts + res_files,
    )
    # On Windows `<name>.<major>.dll` and `<name>.<major>.if.lib`; the release rule renames the latter.
    default_info = DefaultInfo(
        files = depset(dynamic_outputs.files),
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
        "rc_file": attr.label(
            allow_single_file = [".rc"],
            doc = "Windows resource script for the version resource (Make's dll.rc).",
        ),
        "rc_headers": attr.label_list(
            allow_files = True,
            doc = "Headers that `rc_file` includes; their directories are include paths.",
        ),
        "rc_defines": attr.string_list(),
    } | TARGET_ATTRS | VERSION_ATTRS | _CODE_COVERAGE_ATTRS | _PARAMETERS_LIB_ATTRS,
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)


def _cc_exec_impl(ctx):
    if not ctx.attr.deps:
        fail("'{}' has no deps, so there is nothing to link".format(ctx.label))
    toolchain, feature_config = _init_cc_rule(ctx)
    coverage = _coverage_options(ctx, toolchain, feature_config)
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
    runtime_files = _copy_windows_runtime_files(ctx, ctx.files.data)
    default_info = DefaultInfo(
        files = depset([executable] + runtime_files),
        runfiles = ctx.runfiles(files = ctx.files.data + runtime_files).merge_all(
            _collect_runfiles(ctx.attr.deps + ctx.attr.data),
        ),
        executable = executable,
    )
    return [default_info]

_TEST_LINK_MODE_SETTING = str(Label("@config//:test_link_mode"))

def _test_link_mode_transition_impl(settings, attr):
    # Private tests use headers and symbols the release package does not ship.
    if attr.build_from_sources:
        return {_TEST_LINK_MODE_SETTING: "dev"}
    return {_TEST_LINK_MODE_SETTING: settings[_TEST_LINK_MODE_SETTING]}

_test_link_mode_transition = transition(
    implementation = _test_link_mode_transition_impl,
    inputs = [_TEST_LINK_MODE_SETTING],
    outputs = [_TEST_LINK_MODE_SETTING],
)

cc_test = rule(
    implementation = _cc_exec_impl,
    attrs = {
        "lib_tags": attr.string_list(),
        # On the edges, so the test binary and its logs stay in the top-level output directory.
        "deps": attr.label_list(cfg=_test_link_mode_transition),
        "data": attr.label_list(allow_files=True, cfg=_test_link_mode_transition),
        "user_link_flags": attr.string_list(),
        "build_from_sources": attr.bool(default=False),
        "_is_test": attr.bool(default=True),
    } | TARGET_ATTRS | _CODE_COVERAGE_ATTRS,
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
    } | TARGET_ATTRS | _CODE_COVERAGE_ATTRS,
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
    executable = True,
)
