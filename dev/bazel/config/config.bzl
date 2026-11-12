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

load("@onedal//dev/bazel:utils.bzl", "utils", "sets")
load("@onedal//dev/bazel:version.bzl", "DEV_BUILD_DATE", "VERSION")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@onedal//dev/bazel/toolchains:common.bzl", "detect_os", "detect_compiler", "detect_host_arch", "detect_target_arch", "is_dpc_compiler_found")
load("@onedal//dev/bazel/toolchains:cc_toolchain_win.bzl", "win_compiler_id")

ConfigFlagInfo = provider(
    fields = [
        "flag",
        "allowed",
    ],
)

def _config_flag_impl(ctx):
    flag = ctx.build_setting_value
    allowed = ctx.attr.allowed_build_setting_values
    if not flag in allowed:
        fail("Got unexpected value '{}' for {} flag, allowed values are {}".format(
             flag, ctx.attr.name, allowed))
    return ConfigFlagInfo(
        flag = flag,
        allowed = allowed,
    )

_config_flag = rule(
    implementation = _config_flag_impl,
    build_setting = config.string(flag = True),
    attrs = {
        "allowed_build_setting_values": attr.string_list(),
    },
)

def config_flag(name, build_setting_default, allowed_build_setting_values):
    _config_flag(
        name = name,
        build_setting_default = build_setting_default,
        allowed_build_setting_values = allowed_build_setting_values,
    )
    for value in allowed_build_setting_values:
        native.config_setting(
            name = "{}_{}".format(name, value),
            flag_values  = {
                ":" + name: value,
            },
        )

def _config_bool_flag_impl(ctx):
    return ConfigFlagInfo(
        flag = ctx.build_setting_value,
        allowed = [True, False],
    )

config_bool_flag = rule(
    implementation = _config_bool_flag_impl,
    build_setting = config.bool(flag = True),
)

def _unsupported_config_impl(ctx):
    fail(ctx.attr.message)

unsupported_config = rule(
    implementation = _unsupported_config_impl,
    attrs = {
        "message": attr.string(mandatory = True),
    },
)


TargetInfo = provider(
    doc = "The target platform, read once from its constraints by @config//:target.",
    fields = {
        "os": "`lnx`, `win` or `mac`.",
        "arch": "`intel64`, `arm` or `riscv64`.",
        "is_windows": "Whether the target OS is Windows.",
        "is_linux": "Whether the target OS is Linux.",
        "arch_dir": "The `lib/<arch>` directory of the release tree, Make's ARCH_DIR_ONEDAL.",
        "msvc_runtime_debug": "Whether Windows targets use the debug CRT (`--msvc_runtime=debug`).",
    },
)

def _target_info_impl(ctx):
    def has(constraint):
        return ctx.target_platform_has_constraint(constraint[platform_common.ConstraintValueInfo])

    is_windows = has(ctx.attr._windows)
    arch = "arm" if has(ctx.attr._aarch64) else "riscv64" if has(ctx.attr._riscv64) else "intel64"
    return [TargetInfo(
        os = "win" if is_windows else "mac" if has(ctx.attr._osx) else "lnx",
        arch = arch,
        is_windows = is_windows,
        is_linux = has(ctx.attr._linux),
        arch_dir = "ARM64" if arch == "arm" and is_windows else arch,
        msvc_runtime_debug = is_windows and ctx.attr.msvc_runtime[ConfigFlagInfo].flag == "debug",
    )]

target_info = rule(
    implementation = _target_info_impl,
    attrs = {
        "msvc_runtime": attr.label(mandatory = True, providers = [ConfigFlagInfo]),
        "_windows": attr.label(default = "@platforms//os:windows"),
        "_linux": attr.label(default = "@platforms//os:linux"),
        "_osx": attr.label(default = "@platforms//os:osx"),
        "_aarch64": attr.label(default = "@platforms//cpu:aarch64"),
        "_riscv64": attr.label(default = "@platforms//cpu:riscv64"),
    },
)

# Merge into a rule's attrs with `|`; `target(ctx)` then gives the TargetInfo.
TARGET_ATTRS = {
    "_target": attr.label(default = "@config//:target", providers = [TargetInfo]),
}

def target(ctx):
    return ctx.attr._target[TargetInfo]

def validate_build_parameters_lib(value, is_windows):
    if is_windows and value == "yes":
        fail("--build_parameters_lib=yes is not supported on Windows; use auto or no")

def _build_parameters_lib_validation_impl(ctx):
    validate_build_parameters_lib(ctx.attr.flag[ConfigFlagInfo].flag, ctx.attr.target[TargetInfo].is_windows)
    return []

build_parameters_lib_validation = rule(
    implementation = _build_parameters_lib_validation_impl,
    attrs = {
        "flag": attr.label(mandatory = True, providers = [ConfigFlagInfo]),
        "target": attr.label(mandatory = True, providers = [TargetInfo]),
    },
)

CpuInfo = provider(
    fields = [
        "enabled",
        "allowed",
    ],
)

# One row per ISA: its arch, the DAAL dispatch macro, the oneAPI CPU tag and the suffix of the
# per-ISA source files (`_nrh`, `_hsw`, `_skx`); x86 is the only arch with several ISAs.
ISA_TABLE = {
    "sse2": struct(arch = "intel64", dispatch = "SSE2", dal_tag = "SSE2", suffix = "_nrh"),
    "avx2": struct(arch = "intel64", dispatch = "AVX2", dal_tag = "AVX2", suffix = "_hsw"),
    "avx512": struct(arch = "intel64", dispatch = "AVX512", dal_tag = "AVX512", suffix = "_skx"),
    "sve": struct(arch = "arm", dispatch = "A8SVE", dal_tag = "ARMV8SVE", suffix = None),
    "rv64": struct(arch = "riscv64", dispatch = "RV64", dal_tag = "RISCV64GC", suffix = None),
}

_ISA_EXTENSIONS_X86 = [isa for isa, info in ISA_TABLE.items() if info.arch == "intel64"]
_ISA_EXTENSION_AUTO_DEFAULT = "avx2"

def _get_allowed_isa_extensions(ctx):
    # Build settings do not see the target platform, so the arch comes from the repository rule.
    return [isa for isa, info in ISA_TABLE.items() if info.arch == ctx.attr.arch]

def _check_cpu_extensions(extensions, allowed):
    allowed_set = sets.make(allowed)
    requested = sets.make(extensions)
    unsupported = sets.to_list(sets.difference(requested, allowed_set))
    if unsupported:
        fail("Unsupported CPU extensions: {}\n".format(unsupported) +
             "Allowed extensions: {}".format(allowed))

def _cpu_info_impl(ctx):
    allowed = _get_allowed_isa_extensions(ctx)
    is_x86 = allowed == _ISA_EXTENSIONS_X86
    if ctx.build_setting_value in ["all", "modern"]:
        isa_extensions = allowed
    elif ctx.build_setting_value == "auto":
        # The exec host probe means something for a native x86 build only.
        isa_extensions = [ ctx.attr.auto_cpu ] if is_x86 else allowed
    else:
        isa_extensions = ctx.build_setting_value.split(" ")
        isa_extensions = [x.strip() for x in isa_extensions]
    if is_x86:
        # The runtime-dispatch fallback.
        isa_extensions = utils.unique(["sse2"] + isa_extensions)
    else:
        isa_extensions = utils.unique(isa_extensions)
    _check_cpu_extensions(isa_extensions, allowed)
    return CpuInfo(
        enabled = isa_extensions,
        allowed = allowed,
    )

cpu_info = rule(
    implementation = _cpu_info_impl,
    build_setting = config.string(flag = True),
    attrs = {
        "auto_cpu": attr.string(mandatory=True),
        # From detect_target_arch, as this rule's configuration cannot see `--platforms`.
        "arch": attr.string(mandatory=True),
    },
)

VersionInfo = provider(
    fields = [
        "major",
        "minor",
        "update",
        "build",
        "buildrev",
        "status",
        # Binary ABI version, used for the SONAME.
        "binary_major",
        "binary_minor",
    ],
)

def _version_info_impl(ctx):
    build = ctx.attr.build_date[BuildSettingInfo].value or DEV_BUILD_DATE
    if len(build) != 8 or not build.isdigit():
        fail("--build_date must be YYYYMMDD, got '{}'".format(build))
    return [
        VersionInfo(
            major        = ctx.attr.major,
            minor        = ctx.attr.minor,
            update       = ctx.attr.update,
            build        = build,
            buildrev     = ctx.attr.buildrev,
            status       = ctx.attr.status,
            binary_major = ctx.attr.binary_major,
            binary_minor = ctx.attr.binary_minor,
        )
    ]

version_info = rule(
    implementation = _version_info_impl,
    attrs = {
        "major": attr.string(mandatory=True),
        "minor": attr.string(mandatory=True),
        "update": attr.string(mandatory=True),
        "build_date": attr.label(mandatory=True, providers=[BuildSettingInfo]),
        "buildrev": attr.string(mandatory=True),
        "status": attr.string(mandatory=True),
        "binary_major": attr.string(mandatory=True),
        "binary_minor": attr.string(mandatory=True),
    },
)

def _dump_config_info_impl(ctx):
    config_file = ctx.actions.declare_file("config.json")
    flags_json = []
    for target in ctx.attr.flags:
        target_json = "      {}: {},".format(target.label.name, json.encode(target[ConfigFlagInfo]))
        flags_json.append(target_json)
    content = ("{\n" +
    "   cpu: {},\n".format(json.encode(ctx.attr.cpu_info[CpuInfo])) +
    "   version: {},\n".format(json.encode(ctx.attr.version_info[VersionInfo])) +
    "   flags: {\n" +
        "\n".join(flags_json) + "\n" +
    "   }\n" +
    "}\n")
    ctx.actions.write(config_file, content)
    return DefaultInfo(files=depset([ config_file ]))

dump_config_info = rule(
    implementation = _dump_config_info_impl,
    attrs = {
        "cpu_info": attr.label(
            mandatory=True,
            providers = [ CpuInfo ],
        ),
        "version_info": attr.label(
            mandatory=True,
            providers = [ VersionInfo ],
        ),
        "flags": attr.label_list(
            providers = [ ConfigFlagInfo ],
        ),
    },
)

# C++ drivers that pair with the C drivers `detect_compiler` knows.
_CXX_DRIVER = {"gcc": "g++", "clang": "clang++", "icx": "icpx", "icpx": "icpx"}

def _find_cxx_compiler(repo_ctx):
    """The C++ compiler the toolchain will use, for compiling the cpu-detector.

    CXX wins; otherwise the C++ driver that pairs with CC (keeping a version
    suffix or a directory, so `clang-19` gives `clang++-19`); otherwise the
    driver of the compiler the toolchain picks from PATH.
    """
    cxx = repo_ctx.os.environ.get("CXX")
    if cxx:
        return cxx
    cc = repo_ctx.os.environ.get("CC", "")
    for c_driver, cxx_driver in _CXX_DRIVER.items():
        if c_driver in cc:
            return cc.replace(c_driver, cxx_driver, 1)
    return _CXX_DRIVER.get(detect_compiler(repo_ctx, "lnx"), "g++")

def _detect_cpu_extension(repo_ctx):
    cpudetect_src = repo_ctx.path(repo_ctx.attr._cpudetect_src)
    is_windows = "windows" in repo_ctx.os.name
    cpudetect_exe = repo_ctx.path("cpudetect.exe" if is_windows else "cpudetect")
    repo_ctx.report_progress("Compile cpu-detector")
    if is_windows:
        compile_command = [
            "cl", "/nologo", "/EHsc", "/std:c++17",
            cpudetect_src, "/Fe:{}".format(cpudetect_exe),
        ]
    else:
        compile_command = [
            _find_cxx_compiler(repo_ctx), "-pedantic", "-Wall", "-std=c++17",
            cpudetect_src, "-o{}".format(cpudetect_exe),
        ]
    compile_result = repo_ctx.execute(compile_command)
    if compile_result.return_code != 0:
        utils.warn("Cannot compile cpu-detector:\n" +
                   compile_result.stderr + "\n" +
                   "Use {} by default.".format(_ISA_EXTENSION_AUTO_DEFAULT))
        return _ISA_EXTENSION_AUTO_DEFAULT
    repo_ctx.report_progress("Run cpu-detector to determine default vector extension")
    cpudetect_result = repo_ctx.execute([cpudetect_exe])
    if cpudetect_result.return_code != 0:
        utils.warn("Cannot run cpu-detector:\n" +
                   cpudetect_result.stderr + "\n" +
                   "Use {} by default.".format(_ISA_EXTENSION_AUTO_DEFAULT))
        return _ISA_EXTENSION_AUTO_DEFAULT
    return cpudetect_result.stdout.strip()

def _declare_onedal_config_impl(repo_ctx):
    os_id = detect_os(repo_ctx)
    target_arch = detect_target_arch(repo_ctx, detect_host_arch(repo_ctx, os_id))
    # The probe is x86-only; elsewhere `auto_cpu` is ignored.
    auto_cpu = (_detect_cpu_extension(repo_ctx) if target_arch == "intel64"
                else _ISA_EXTENSION_AUTO_DEFAULT)

    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/config:config.tpl.BUILD"),
        substitutions = {
            "%{auto_cpu}":              auto_cpu,
            "%{arch}":                  target_arch,
            "%{backend_config_default}": "mkl" if target_arch == "intel64" else "ref",
            # On Windows only the icx toolchain has a DPC++ driver.
            "%{dpc_compiler_found}":    str(win_compiler_id(repo_ctx, target_arch) == "icx"
                                            if os_id == "win" else is_dpc_compiler_found(repo_ctx, os_id)),
            "%{version_major}":         VERSION.major,
            "%{version_minor}":         VERSION.minor,
            "%{version_update}":        VERSION.update,
            "%{version_buildrev}":      "work",
            "%{version_status}":        VERSION.status,
            "%{version_binary_major}":  VERSION.binary_major,
            "%{version_binary_minor}":  VERSION.binary_minor,
        },
    )
declare_onedal_config = repository_rule(
    implementation = _declare_onedal_config_impl,
    local = True,
    environ = [
        "CC",
        "CXX",
        # PATH, so that sourcing setvars.sh re-detects the DPC++ compiler.
        "PATH",
        # The Windows compiler, which decides the --release_dpc default there.
        "ONEDAL_WIN_COMPILER",
        # Windows host-arch detection (see toolchains/common.bzl).
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_ARCHITEW6432",
    ],
    attrs = {
        "_cpudetect_src": attr.label(
            default = "@onedal//dev/bazel/config:cpudetect.cpp",
        ),
    },
)
