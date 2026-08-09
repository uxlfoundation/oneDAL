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

load("@onedal//dev/bazel/config:config.bzl", "ConfigFlagInfo", "VersionInfo")

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
            "@ARCH_DIR_ONEDAL@": "intel64",
            "@DLL_REL_PATH@": "redist",
            "@INC_REL_PATH@": "include",
            "@oneDAL_VERSION@": "",
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
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
    },
)

def generate_cmake_config(name, template, out, **kwargs):
    _generate_cmake_config(
        name = name,
        template = template,
        out = out,
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

    if ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo]):
        # A debug MSVC runtime renames every oneDAL library (`onedal_cored.lib`)
        # and switches both the CRT flag and the TBB variant, so the .pc file has
        # to follow. See `_msvc_runtime_suffix` in dev/bazel/cc.bzl.
        is_rt_debug = ctx.attr._msvc_runtime[ConfigFlagInfo].flag == "debug"
        d = "d" if is_rt_debug else ""
        # Keep this list in sync with the `_WIN32` branch of
        # `deploy/pkg-config/pkg-config.cpp`: the dynamic package exposes only
        # onedal and onedal_core, and Windows has no parameters library at all
        # (`makefile` errors out on `BUILD_PARAMETERS_LIB=yes` there).
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
libdir=${{exec_prefix}}/lib/intel64
includedir=${{prefix}}/include

Name: oneDAL
Description: oneAPI Data Analytics Library
Version: {major}.{minor}
URL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html
Libs: {onedal_libs} mkl_core.lib mkl_intel_lp64.lib mkl_tbb_thread.lib tbb12{tbb_d}.lib tbbmalloc{tbb_d}.lib
Cflags: /std:c++17 {crt} /wd4996 /EHsc -I${{includedir}}
""".format(
                major = vi.major,
                minor = vi.minor,
                onedal_libs = onedal_libs,
                crt = "/MDd" if is_rt_debug else "/MD",
                tbb_d = "_debug" if is_rt_debug else "",
            ),
        )
    else:
        suffix = "a" if ctx.attr.static else "so"
        ctx.actions.write(
            output = out,
            content = _PKGCONFIG_LICENSE_HEADER + """prefix=${{pcfiledir}}/../../
exec_prefix=${{prefix}}
libdir=${{exec_prefix}}/lib/intel64
includedir=${{prefix}}/include

Name: oneDAL
Description: oneAPI Data Analytics Library
Version: {major}.{minor}
URL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html
Libs: ${{libdir}}/libonedal.{suffix} ${{libdir}}/libonedal_core.{suffix} ${{libdir}}/libonedal_thread.{suffix} ${{libdir}}/libonedal_parameters.{suffix} -lmkl_core -lmkl_intel_lp64 -lmkl_tbb_thread -ltbb -ltbbmalloc -lpthread -ldl
Cflags: -std=c++17 -Wno-deprecated-declarations -I${{includedir}}
""".format(
                major = vi.major,
                minor = vi.minor,
                suffix = suffix,
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
    },
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
