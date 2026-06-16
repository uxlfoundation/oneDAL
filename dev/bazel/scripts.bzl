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

load("@onedal//dev/bazel/config:config.bzl", "VersionInfo")

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

def _generate_pkgconfig_impl(ctx):
    """Generate pkg-config files matching deploy/pkg-config/pkg-config.cpp."""
    vi = ctx.attr._version_info[VersionInfo]
    out = ctx.actions.declare_file(ctx.attr.out)

    if ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo]):
        onedal_libs = (
            "${libdir}/onedal.lib ${libdir}/onedal_core.lib ${libdir}/onedal_thread.lib"
            if ctx.attr.static else
            "${libdir}/onedal_dll.lib ${libdir}/onedal_parameters_dll.lib ${libdir}/onedal_core_dll.lib"
        )
        ctx.actions.write(
            output = out,
            content = """prefix=${{pcfiledir}}/../../
exec_prefix=${{prefix}}
libdir=${{exec_prefix}}/lib/intel64
includedir=${{prefix}}/include

Name: oneDAL
Description: oneAPI Data Analytics Library
Version: {major}.{minor}
URL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onedal.html
Libs: {onedal_libs} mkl_core.lib mkl_intel_ilp64.lib mkl_tbb_thread.lib tbb12.lib tbbmalloc.lib
Cflags: /std:c++17 /MD /wd4996 /EHsc -I${{includedir}}
""".format(
                major = vi.major,
                minor = vi.minor,
                onedal_libs = onedal_libs,
            ),
        )
    else:
        suffix = "a" if ctx.attr.static else "so"
        ctx.actions.write(
            output = out,
            content = """#===============================================================================
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
prefix=${{pcfiledir}}/../../
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
