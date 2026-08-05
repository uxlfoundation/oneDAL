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

load("@onedal//dev/bazel:cc.bzl",
    "cc_module",
    "cc_static_lib",
    "cc_dynamic_lib",
)
load("@onedal//dev/bazel:utils.bzl",
    "sets",
    "paths",
)
load("@onedal//dev/bazel/config:config.bzl",
    "CpuInfo",
    "VersionInfo",
)

def daal_module(name, features=[], lib_tag="daal",
                hdrs=[], srcs=[], auto=False,
                local_defines=[], copts=[], visibility_hidden=True, **kwargs):
    if auto:
        auto_hdrs = native.glob(["**/*.h", "**/*.i"], allow_empty=True,)
        auto_srcs = native.glob(["**/*.cpp"], allow_empty=True,)
    else:
        auto_hdrs = []
        auto_srcs = []
    cc_module(
        name = name,
        lib_tag = lib_tag,
        features = [ "c++17" ] + features,
        cpu_defines = {
            "sse2":       [ "DAAL_CPU=sse2"      ],
            "avx2":       [ "DAAL_CPU=avx2"       ],
            "avx512":     [ "DAAL_CPU=avx512"     ],
        },
        fpt_defines = {
            "f32": [ "DAAL_FPTYPE=float"  ],
            "f64": [ "DAAL_FPTYPE=double" ],
        },
        hdrs = auto_hdrs + hdrs,
        srcs = auto_srcs + srcs,
        copts = copts + (select({
            "@platforms//os:windows": ["/utf-8"],
            "//conditions:default": ["-fvisibility=hidden", "-fvisibility-inlines-hidden"],
        }) if visibility_hidden else select({
            "@platforms//os:windows": ["/utf-8"],
            "//conditions:default": [],
        })),
        local_defines = select({
            "@config//:assert_enabled": local_defines + ["__DAAL_IMPLEMENTATION", "DEBUG_ASSERT=1"],
            "//conditions:default": local_defines + ["__DAAL_IMPLEMENTATION"],
        }) + select({
            "@config//:backend_ref": [
                "DAAL_REF",
            ],
            "//conditions:default": [],
        }),
        **kwargs,
    )

def daal_static_lib(name, lib_tags=["daal"], **kwargs):
    cc_static_lib(
        name = name,
        lib_tags = lib_tags,
        **kwargs,
    )

_MKL_EXCLUDE_LIBS_FLAGS = select({
    # Hide MKL static symbols from the dynamic lib's export table.
    # --exclude-libs hides all symbols from the named archive, ensuring that
    # MKL objects embedded via --whole-archive are not re-exported.
    # This matches Make's behaviour (see dev/make/deps.mk MKL linkage).
    # GNU ld only; not supported on Windows (MSVC) or macOS (Apple ld).
    "@config//:backend_config_mkl_linux": [
        "-Wl,--exclude-libs=libmkl_tbb_thread.a",
        "-Wl,--exclude-libs=libmkl_core.a",
        "-Wl,--exclude-libs=libmkl_intel_ilp64.a",
    ],
    "//conditions:default": [],
})

def daal_dynamic_lib(name, lib_tags=["daal", "mkl_embed"], **kwargs):
    linkopts = kwargs.pop("linkopts", [])
    cc_dynamic_lib(
        name = name,
        lib_tags = lib_tags,
        linkopts = linkopts + _MKL_EXCLUDE_LIBS_FLAGS,
        **kwargs,
    )

def daal_algorithms(name, algorithms=[]):
    alg_labels = []
    for alg_name in algorithms:
        label = "@onedal//cpp/daal/src/algorithms/{}:kernel".format(alg_name)
        alg_labels.append(label)
    cc_module(
        name = name,
        deps = alg_labels,
    )

def _daal_generate_version_impl(ctx):
    vi = ctx.attr._version_info[VersionInfo]
    version = ctx.actions.declare_file(ctx.attr.out)
    content = (
        "// DO NOT EDIT: file is auto-generated on build time\n" +
        "// DO NOT PUT THIS FILE TO SVC: file is auto-generated on build time\n" +
        "// Product version is specified in dev/bazel/config.bzl file\n" +
        "\n" +
        "#define MAJORVERSION {}\n".format(vi.major) +
        "#define MINORVERSION {}\n".format(vi.minor) +
        "#define UPDATEVERSION {}\n".format(vi.update) +
        "#define BUILD \"{}\"\n".format(vi.build) +
        "#define BUILD_REV \"{}\"\n".format(vi.buildrev) +
        "#define PRODUCT_STATUS '{}'\n".format(vi.status)
    )
    ctx.actions.write(version, content)
    return [ DefaultInfo(files=depset([ version ])) ]

daal_generate_version = rule(
    implementation = _daal_generate_version_impl,
    output_to_genfiles = True,
    attrs = {
        "out": attr.string(mandatory=True),
        "_version_info": attr.label(
            default = "@config//:version",
        ),
    },
)

def _get_disabled_cpus(ctx):
    cpu_info = ctx.attr._cpus[CpuInfo]
    all_cpus = sets.make(cpu_info.allowed)
    enabled_cpus = sets.make(cpu_info.enabled)
    return sets.difference(all_cpus, enabled_cpus)

def _declare_patched_kernel_defines(ctx):
    relpath = paths.dirname(ctx.build_file_path)
    patched_path = paths.relativize(ctx.file.src.path, relpath)
    return ctx.actions.declare_file(patched_path)

def _daal_patch_kernel_defines_impl(ctx):
    """Strip `#define DAAL_KERNEL_<ISA>` lines for disabled CPUs.

    Previously handled by three shell/cmd/ps1 tools registered via a
    dedicated `extra_toolchain`. Doing the substitution with
    `expand_template` moves it into pure Starlark: no shell, no platform
    branch, no toolchain wiring.
    """
    disabled_cpus = sets.to_list(_get_disabled_cpus(ctx))
    substitutions = {
        "#define DAAL_KERNEL_{}\n".format(cpu.upper()): ""
        for cpu in disabled_cpus
    }
    patched = _declare_patched_kernel_defines(ctx)
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = patched,
        substitutions = substitutions,
    )
    return [ DefaultInfo(files=depset([patched])) ]

daal_patch_kernel_defines = rule(
    implementation = _daal_patch_kernel_defines_impl,
    output_to_genfiles = True,
    attrs = {
        "src": attr.label(allow_single_file=True, mandatory=True),
        "_cpus": attr.label(
            default = "@config//:cpu",
        ),
    },
)
