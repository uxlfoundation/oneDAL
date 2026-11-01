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

"""Rules for generating release scripts and configuration files."""

load("@onedal//dev/bazel/config:config.bzl", "ConfigFlagInfo", "RELEASE_ARCH_ATTRS", "VersionInfo", "release_arch_dir")

# ---------------------------------------------------------------------------
# Versioned template files
# ---------------------------------------------------------------------------

def _generate_versioned_template_impl(ctx):
    """Expand a release template substituting binary version placeholders."""
    vi = ctx.attr._version_info[VersionInfo]
    out = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = out,
        substitutions = {
            "__DAL_MAJOR_BINARY__": vi.binary_major,
            "__DAL_MINOR_BINARY__": vi.binary_minor,
        },
    )
    return [DefaultInfo(files = depset([out]))]

_generate_versioned_template = rule(
    implementation = _generate_versioned_template_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "Source release template file.",
        ),
        "out": attr.string(
            mandatory = True,
            doc = "Output path relative to the package.",
        ),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
    },
)

def generate_vars_sh(name, out = "env/vars.sh", **kwargs):
    """Generate release environment script from template."""
    _generate_versioned_template(
        name = name,
        template = select({
            "@platforms//os:windows": "@onedal//deploy/local:vars_win.bat",
            "@platforms//os:osx": "@onedal//deploy/local:vars_mac.sh",
            "//conditions:default": "@onedal//deploy/local:vars_lnx.sh",
        }),
        out = out,
        **kwargs
    )

def generate_modulefile(name, out = "modulefiles/dal", **kwargs):
    """Generate Linux modulefile from template."""
    _generate_versioned_template(
        name = name,
        template = "@onedal//deploy/local:dal",
        out = out,
        **kwargs
    )

# ---------------------------------------------------------------------------
# CMake package config
# ---------------------------------------------------------------------------

def _generate_cmake_config_impl(ctx):
    vi = ctx.attr._version_info[VersionInfo]
    out = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = out,
        substitutions = {
            "@DAL_ROOT_REL_PATH@": "../../..",
            "@VERSIONS_SET@": "FALSE",
            "@DAL_VER_MAJOR_BIN@": "",
            "@DAL_VER_MINOR_BIN@": "",
            "@ARCH_DIR_ONEDAL@": release_arch_dir(ctx),
            "@DLL_REL_PATH@": "redist",
            "@INC_REL_PATH@": "include",
            "@oneDAL_VERSION@": "",
            "@ONEDAL_USE_PARAMETERS_LIBRARY@": "yes" if ctx.attr.parameters_lib else "no",
        },
    )
    return [DefaultInfo(files = depset([out]))]

_generate_cmake_config = rule(
    implementation = _generate_cmake_config_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "out": attr.string(mandatory = True),
        "parameters_lib": attr.bool(default = True),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
    } | RELEASE_ARCH_ATTRS,
)

def generate_cmake_config(name, template, out, **kwargs):
    _generate_cmake_config(
        name = name,
        template = template,
        out = out,
        **kwargs
    )

# ---------------------------------------------------------------------------
# NuGet nuspec
# ---------------------------------------------------------------------------

# Content summary per distribution type, as in deploy/nuget/prepare_dal_nuget.sh.
_NUSPEC_CONTENT = {
    "redist": "dynamic libraries and headers",
    "devel": "dynamic and static libraries and headers",
    "static": "static libraries and headers",
}

def _generate_nuspec_impl(ctx):
    """Expand deploy/nuget/inteldal.nuspec.tpl as prepare_dal_nuget.sh does.

    The script maps the Linux platform to `linux` (TBB `linux`) and the Windows
    platform to `win-x64` (TBB `win`), then runs one `sed` over the template.
    Both the Linux and the Windows file come from this rule; `release_extra_file`
    picks which one to stage on which OS.
    """
    vi = ctx.attr._version_info[VersionInfo]
    # identify_os.sh gives mac32e on macOS, which prepare_dal_nuget.sh maps to
    # osx-x64 / osx (its `elif` branch for mac32e).
    if ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo]):
        platform, tbb_platform = "win-x64", "win"
    elif ctx.target_platform_has_constraint(ctx.attr._osx_constraint[platform_common.ConstraintValueInfo]):
        platform, tbb_platform = "osx-x64", "osx"
    else:
        platform, tbb_platform = "linux", "linux"
    out = ctx.actions.declare_file("inteldal.{}.nuspec".format(ctx.attr.distr_type))
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = out,
        substitutions = {
            "__DISTRTYPE__": ctx.attr.distr_type,
            "__PLATFORM__": platform,
            "__TBB_PLATFORM__": tbb_platform,
            # prepare_dal_nuget.sh builds the version from the header's
            # major, minor and update numbers.
            "__VERSION__": "{}.{}.{}".format(vi.major, vi.minor, vi.update),
            "__CONTENT__": _NUSPEC_CONTENT[ctx.attr.distr_type],
        },
    )
    return [DefaultInfo(files = depset([out]))]

_generate_nuspec = rule(
    implementation = _generate_nuspec_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "distr_type": attr.string(
            mandatory = True,
            values = ["redist", "devel", "static"],
        ),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
        "_osx_constraint": attr.label(
            default = "@platforms//os:osx",
        ),
    },
)

def generate_nuspec(name, template, distr_type, **kwargs):
    _generate_nuspec(
        name = name,
        template = template,
        distr_type = distr_type,
        **kwargs
    )

# ---------------------------------------------------------------------------
# pkg-config
# ---------------------------------------------------------------------------

# Make builds the .pc files by running the C preprocessor over
# `deploy/pkg-config/pkg-config.cpp`, prepending that file's first 16 lines and
# rewriting `//` into `#` (see the `_release_c` recipe in `makefile`). The result
# is this 15-line header on every platform, so both branches below emit it.
_PKGCONFIG_LICENSE_HEADER = """#===============================================================================
# Copyright contributors to the oneDAL Project
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
"""

def _generate_pkgconfig_impl(ctx):
    """Generate pkg-config files matching deploy/pkg-config/pkg-config.cpp."""
    vi = ctx.attr._version_info[VersionInfo]
    out = ctx.actions.declare_file(ctx.attr.out)
    is_windows = ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo])
    arch_dir = release_arch_dir(ctx)

    if is_windows:
        # A debug MSVC runtime renames every oneDAL library (`onedal_cored.lib`)
        # and switches both the CRT flag and the TBB variant, so the .pc file has
        # to follow. See `_msvc_runtime_suffix` in dev/bazel/cc.bzl.
        is_rt_debug = ctx.attr._msvc_runtime[ConfigFlagInfo].flag == "debug"
        d = "d" if is_rt_debug else ""
        # What has to stay in sync with the `_WIN32` branch of
        # `deploy/pkg-config/pkg-config.cpp` is *which* libraries appear: the
        # dynamic package exposes only onedal and onedal_core, and Windows has no
        # parameters library at all (`makefile` errors out on
        # `BUILD_PARAMETERS_LIB=yes` there). The names themselves are not a 1:1
        # match and are not meant to be -- the deploy-side generator knows nothing
        # about the MSVC runtime, while this one carries the `d` suffix, `/MDd`
        # and the debug TBB variant. Only the set of entries is common.
        #
        # The braces around `libdir` are doubled because this string is formatted
        # here, for `{d}`; it then goes into the template below as an argument,
        # where `format` substitutes it without looking at it again.
        onedal_libs = (
            "${{libdir}}/onedal{d}.lib ${{libdir}}/onedal_core{d}.lib ${{libdir}}/onedal_thread{d}.lib"
            if ctx.attr.static else
            "${{libdir}}/onedal{d}_dll.lib ${{libdir}}/onedal_core{d}_dll.lib"
        ).format(d = d)
        ctx.actions.write(
            output = out,
            content = _PKGCONFIG_LICENSE_HEADER + """prefix=${{pcfiledir}}/../../
exec_prefix=${{prefix}}
libdir=${{exec_prefix}}/lib/{arch_dir}
includedir=${{prefix}}/include

Name: oneDAL
Description: oneAPI Data Analytics Library
Version: {major}.{minor}
URL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html
Libs: {onedal_libs} mkl_core.lib mkl_intel_lp64.lib mkl_tbb_thread{mkl_d}.lib tbb12{tbb_d}.lib tbbmalloc{tbb_d}.lib
Cflags: /std:c++17 {crt} /wd4996 /EHsc -I${{includedir}}
""".format(
                arch_dir = arch_dir,
                major = vi.major,
                minor = vi.minor,
                onedal_libs = onedal_libs,
                crt = "/MDd" if is_rt_debug else "/MD",
                tbb_d = "_debug" if is_rt_debug else "",
                # Only MKL's threading layer has a debug-CRT variant, and it
                # uses a plain `d` (dev/make/deps.mkl.mk:51); mkl_core and
                # mkl_intel_lp64 are unsuffixed in both flavours.
                mkl_d = "d" if is_rt_debug else "",
            ),
        )
    else:
        suffix = "a" if ctx.attr.static else "so"

        # Keep this in sync with the `__linux__` branch of
        # `deploy/pkg-config/pkg-config.cpp`: the static libraries are built
        # against the ILP64 oneMKL interface, the dynamic ones against LP64, and
        # `libonedal_parameters` exists only in the separate layout. oneDAL
        # links the oneMKL TBB threading layer, so no OpenMP runtime belongs on
        # a host consumer's link line.
        #
        # The static package names the oneMKL *archives* with `-l:` inside a
        # link group: `-lmkl_core` finds `libmkl_core.so`, whose internals live
        # in the kernel libraries oneMKL dlopen's at runtime, so a static
        # consumer linking the shared libraries gets ~9600 undefined
        # references. The archives alone are not enough either -- they are
        # mutually recursive -- and the oneDAL archives sit inside the same
        # group so the link does not depend on their order.
        # Single braces: these are `format()` arguments, inserted verbatim.
        parameters_lib = " ${libdir}/libonedal_parameters." + suffix if ctx.attr.parameters_lib else ""
        onedal_libs = "${{libdir}}/libonedal.{0} ${{libdir}}/libonedal_core.{0} ${{libdir}}/libonedal_thread.{0}{1}".format(
            suffix,
            parameters_lib,
        )
        if ctx.attr.static:
            onedal_libs = ("-Wl,--start-group " + onedal_libs +
                           " -l:libmkl_intel_ilp64.a -l:libmkl_tbb_thread.a -l:libmkl_core.a" +
                           " -Wl,--end-group -ltbb -ltbbmalloc -lpthread -ldl")
        else:
            onedal_libs += " -lmkl_core -lmkl_intel_lp64 -lmkl_tbb_thread -ltbb -ltbbmalloc -lpthread -ldl"
        ctx.actions.write(
            output = out,
            content = _PKGCONFIG_LICENSE_HEADER + """prefix=${{pcfiledir}}/../../
exec_prefix=${{prefix}}
libdir=${{exec_prefix}}/lib/{arch_dir}
includedir=${{prefix}}/include

Name: oneDAL
Description: oneAPI Data Analytics Library
Version: {major}.{minor}
URL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html
Libs: {onedal_libs}
Cflags: -std=c++17 -Wno-deprecated-declarations -I${{includedir}}
""".format(
                arch_dir = arch_dir,
                major = vi.major,
                minor = vi.minor,
                onedal_libs = onedal_libs,
            ),
        )
    return [DefaultInfo(files = depset([out]))]

_generate_pkgconfig = rule(
    implementation = _generate_pkgconfig_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "pkg-config.cpp template file.",
        ),
        "out": attr.string(
            mandatory = True,
            doc = "Output path relative to the package.",
        ),
        "static": attr.bool(
            default = False,
            doc = "Generate a static-library pkg-config file.",
        ),
        "parameters_lib": attr.bool(default = True),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
        "_msvc_runtime": attr.label(
            default = "@config//:msvc_runtime",
            providers = [ConfigFlagInfo],
        ),
    } | RELEASE_ARCH_ATTRS,
)

def generate_pkgconfig(name, out = "lib/pkgconfig/onedal.pc", static = False, **kwargs):
    """Generate pkg-config .pc file from deploy/pkg-config/pkg-config.cpp."""
    _generate_pkgconfig(
        name = name,
        template = "@onedal//deploy/pkg-config:pkg-config.cpp",
        out = out,
        static = static,
        **kwargs
    )
