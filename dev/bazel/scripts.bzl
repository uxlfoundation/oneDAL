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

load("@onedal//dev/bazel/config:config.bzl", "TARGET_ATTRS", "VERSION_ATTRS", "get_version_info", "target")

# --- Versioned template files ---

def _generate_versioned_template_impl(ctx):
    """Expand a release template substituting binary version placeholders."""
    vi = get_version_info(ctx)
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
    } | VERSION_ATTRS,
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

# --- CMake package config ---

def _generate_cmake_config_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.out)
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = out,
        substitutions = {
            "@DAL_ROOT_REL_PATH@": "../../..",
            "@VERSIONS_SET@": "FALSE",
            "@DAL_VER_MAJOR_BIN@": "",
            "@DAL_VER_MINOR_BIN@": "",
            "@ARCH_DIR_ONEDAL@": target(ctx).arch_dir,
            "@DLL_REL_PATH@": "redist",
            "@INC_REL_PATH@": "include",
            "@oneDAL_VERSION@": "",
            "@ONEDAL_USE_PARAMETERS_LIBRARY@": "yes" if ctx.attr.parameters_lib else "no",
        },
    )
    return [DefaultInfo(files = depset([out]))]

generate_cmake_config = rule(
    implementation = _generate_cmake_config_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "out": attr.string(mandatory = True),
        "parameters_lib": attr.bool(default = True),
    } | TARGET_ATTRS,
)

# --- NuGet nuspec ---

# Content summary per distribution type, as in deploy/nuget/prepare_dal_nuget.sh.
_NUSPEC_CONTENT = {
    "redist": "dynamic libraries and headers",
    "devel": "dynamic and static libraries and headers",
    "static": "static libraries and headers",
}

def _generate_nuspec_impl(ctx):
    """Expand deploy/nuget/inteldal.nuspec.tpl as prepare_dal_nuget.sh does."""
    vi = get_version_info(ctx)
    t = target(ctx)
    # Same ids as prepare_dal_nuget.sh: x64 keeps linux, win-x64, osx-x64 (Make has no macOS ARM64).
    # ARM64 and RISC-V get their own ids, so they never publish under an x64 name.
    if t.is_windows:
        platform, tbb_platform = "win-arm64" if t.arch == "arm" else "win-x64", "win"
    elif t.os == "mac":
        platform, tbb_platform = "osx-x64", "osx"
    else:
        platform, tbb_platform = {"arm": "linux-arm64", "riscv64": "linux-riscv64"}.get(t.arch, "linux"), "linux"
    out = ctx.actions.declare_file("inteldal.{}.nuspec".format(ctx.attr.distr_type))
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = out,
        substitutions = {
            "__DISTRTYPE__": ctx.attr.distr_type,
            "__PLATFORM__": platform,
            "__TBB_PLATFORM__": tbb_platform,
            "__VERSION__": "{}.{}.{}".format(vi.major, vi.minor, vi.update),
            "__CONTENT__": _NUSPEC_CONTENT[ctx.attr.distr_type],
        },
    )
    return [DefaultInfo(files = depset([out]))]

generate_nuspec = rule(
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
    } | TARGET_ATTRS | VERSION_ATTRS,
)

# --- pkg-config ---

# The header Make's `_release_c` recipe copies from pkg-config.cpp into every .pc file.
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
    """Write a pkg-config file with the content of deploy/pkg-config/pkg-config.cpp."""
    vi = get_version_info(ctx)
    out = ctx.actions.declare_file(ctx.attr.out)
    t = target(ctx)
    arch_dir = t.arch_dir

    if t.is_windows:
        # The debug CRT renames the oneDAL libraries and switches the CRT flag and TBB variant.
        is_rt_debug = t.msvc_runtime_debug
        d = "d" if is_rt_debug else ""
        # The same libraries as the `_WIN32` branch of pkg-config.cpp. Doubled braces survive
        # this format(); the template's format() inserts the result verbatim.
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
                # Only MKL's threading layer has a debug-CRT variant.
                mkl_d = "d" if is_rt_debug else "",
            ),
        )
    else:
        suffix = "a" if ctx.attr.static else "so"

        # Keep in sync with the `__linux__` branch of pkg-config.cpp. Static links name the
        # mutually recursive oneMKL archives in one group, as `-lmkl_core` would pick the .so.

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
        "out": attr.string(
            mandatory = True,
            doc = "Output path relative to the package.",
        ),
        "static": attr.bool(
            default = False,
            doc = "Generate a static-library pkg-config file.",
        ),
        "parameters_lib": attr.bool(default = True),
    } | TARGET_ATTRS | VERSION_ATTRS,
)

def generate_pkgconfig(name, out, static = False, **kwargs):
    _generate_pkgconfig(
        name = name,
        out = out,
        static = static,
        **kwargs
    )
