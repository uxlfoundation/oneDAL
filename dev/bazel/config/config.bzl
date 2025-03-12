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

CpuInfo = provider(
    fields = [
        "enabled",
        "allowed",
    ],
)

_ISA_EXTENSIONS = ["sse2", "sse42", "avx2", "avx512"]
_ISA_EXTENSIONS_MODERN = ["sse2", "avx2", "avx512"]
_ISA_EXTENSION_AUTO_DEFAULT = "avx2"

def _check_cpu_extensions(extensions):
    allowed = sets.make(_ISA_EXTENSIONS)
    requested = sets.make(extensions)
    unsupported = sets.to_list(sets.difference(requested, allowed))
    if unsupported:
        fail("Unsupported CPU extensions: {}\n".format(unsupported) +
             "Allowed extensions: {}".format(_ISA_EXTENSIONS))

def _cpu_info_impl(ctx):
    if ctx.build_setting_value == "all":
        isa_extensions = _ISA_EXTENSIONS
    elif ctx.build_setting_value == "modern":
        isa_extensions = _ISA_EXTENSIONS_MODERN
    elif ctx.build_setting_value == "auto":
        isa_extensions = [ ctx.attr.auto_cpu ]
    else:
        isa_extensions = ctx.build_setting_value.split(" ")
        isa_extensions = [x.strip() for x in isa_extensions]
    isa_extensions = utils.unique(["sse2"] + isa_extensions)
    _check_cpu_extensions(isa_extensions)
    return CpuInfo(
        enabled = isa_extensions,
        allowed = _ISA_EXTENSIONS,
    )

cpu_info = rule(
    implementation = _cpu_info_impl,
    build_setting = config.string(flag = True),
    attrs = {
        "auto_cpu": attr.string(mandatory=True),
        "verbose": attr.label(),
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
    ],
)

def _version_info_impl(ctx):
    return [
        VersionInfo(
            major    = ctx.attr.major,
            minor    = ctx.attr.minor,
            update   = ctx.attr.update,
            build    = ctx.attr.build,
            buildrev = ctx.attr.buildrev,
            status   = ctx.attr.status,
        )
    ]

version_info = rule(
    implementation = _version_info_impl,
    attrs = {
        "major": attr.string(mandatory=True),
        "minor": attr.string(mandatory=True),
        "update": attr.string(mandatory=True),
        "build": attr.string(mandatory=True),
        "buildrev": attr.string(mandatory=True),
        "status": attr.string(mandatory=True),
    },
)

def _dump_config_info_impl(ctx):
    config_file = ctx.actions.declare_file("config.json")
    flags_json = []
    for target in ctx.attr.flags:
        json = "      {}: {},".format(target.label.name, target[ConfigFlagInfo].to_json())
        flags_json.append(json)
    content = ("{\n" +
    "   cpu: {},\n".format(ctx.attr.cpu_info[CpuInfo].to_json()) +
    "   version: {},\n".format(ctx.attr.version_info[VersionInfo].to_json()) +
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

def _detect_cpu_extension(repo_ctx):
    cpudetect_src = repo_ctx.path(repo_ctx.attr._cpudetect_src)
    cpudetect_exe = repo_ctx.path("cpudetect")
    repo_ctx.report_progress("Compile cpu-detector")
    compile_result = repo_ctx.execute([
        "g++", "-pedantic", "-Wall", "-std=c++11",
        cpudetect_src, "-o{}".format(cpudetect_exe),
    ])
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
    auto_cpu = _detect_cpu_extension(repo_ctx)
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/config:config.tpl.BUILD"),
        substitutions = {
            "%{auto_cpu}":         auto_cpu,
            "%{version_major}":    "2025",
            "%{version_minor}":    "2",
            "%{version_update}":   "0",
            "%{version_build}":    utils.datestamp(repo_ctx),
            "%{version_buildrev}": "work",
            "%{version_status}":   "P",
        },
    )

declare_onedal_config = repository_rule(
    implementation = _declare_onedal_config_impl,
    local = True,
    attrs = {
        "_cpudetect_src": attr.label(
            default = "@onedal//dev/bazel/config:cpudetect.cpp",
        ),
    },
)
