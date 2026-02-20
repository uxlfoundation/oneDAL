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

"""Rules for generating release scripts and configuration files.

Provides:
  - generate_vars_sh: generates env/vars.sh (Linux/macOS) from a template
  - generate_pkgconfig: generates lib/pkgconfig/onedal.pc via cpp preprocessor
"""

load("@onedal//dev/bazel/config:config.bzl", "VersionInfo")

# ---------------------------------------------------------------------------
# vars.sh
# ---------------------------------------------------------------------------

def _generate_vars_sh_impl(ctx):
    """Expand vars.sh template substituting binary version placeholders."""
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

_generate_vars_sh = rule(
    implementation = _generate_vars_sh_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "Source vars.sh template file (e.g. deploy/local/vars_lnx.sh).",
        ),
        "out": attr.string(
            mandatory = True,
            doc = "Output path relative to the package (e.g. 'env/vars.sh').",
        ),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
    },
)

def generate_vars_sh(name, out = "env/vars.sh", **kwargs):
    """Generate release environment script from template.

    Substitutes __DAL_MAJOR_BINARY__ and __DAL_MINOR_BINARY__ with the
    binary ABI version numbers defined in config.bzl.

    Args:
        name: Target name.
        out:  Output path (default: 'env/vars.sh').
    """
    _generate_vars_sh(
        name = name,
        template = select({
            # TODO: add Windows condition when Windows toolchain is ready.
            # "@platforms//os:windows": "@onedal//deploy/local:vars_win.bat",
            "//conditions:default": "@onedal//deploy/local:vars_lnx.sh",
        }),
        out = out,
        **kwargs
    )

# ---------------------------------------------------------------------------
# pkg-config
# ---------------------------------------------------------------------------

def _generate_pkgconfig_impl(ctx):
    """Generate onedal.pc via the C preprocessor from pkg-config.cpp template.

    Note: deploy/pkg-config/pkg-config.cpp currently hardcodes Version: 2026.0
    and does not use DAL_MAJOR/MINOR defines. The -D flags below are passed for
    future compatibility if the template is updated to use them.
    """
    vi = ctx.attr._version_info[VersionInfo]
    out = ctx.actions.declare_file(ctx.attr.out)

    # pkg-config.cpp uses #if/#define blocks and the cpp preprocessor
    # to emit different content for static vs dynamic variants.
    # expand_template is not used here because the template relies on
    # cpp's conditional compilation, which cannot be replicated with
    # simple string substitution.
    # Use gcc -E -P (C preprocessor) instead of cpp directly.
    # 'cpp' may not find cc1plus in some CI environments; 'gcc -E -P' is more portable.
    ctx.actions.run_shell(
        inputs = [ctx.file.template],
        outputs = [out],
        command = (
            "gcc -E -P -x c " +
            "-DDAL_MAJOR_BINARY={binary_major} " +
            "-DDAL_MINOR_BINARY={binary_minor} " +
            "-DDAL_MAJOR={major} " +
            "-DDAL_MINOR={minor} " +
            "{template} -o {out}"
        ).format(
            binary_major = vi.binary_major,
            binary_minor = vi.binary_minor,
            major = vi.major,
            minor = vi.minor,
            template = ctx.file.template.path,
            out = out.path,
        ),
        mnemonic = "GenPkgConfig",
        progress_message = "Generating pkg-config file {}".format(out.short_path),
        use_default_shell_env = True,
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
            doc = "Output path relative to the package (e.g. 'lib/pkgconfig/onedal.pc').",
        ),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
    },
)

def generate_pkgconfig(name, out = "lib/pkgconfig/onedal.pc", **kwargs):
    """Generate pkg-config .pc file from deploy/pkg-config/pkg-config.cpp.

    Args:
        name: Target name.
        out:  Output path (default: 'lib/pkgconfig/onedal.pc').
    """
    _generate_pkgconfig(
        name = name,
        template = "@onedal//deploy/pkg-config:pkg-config.cpp",
        out = out,
        **kwargs
    )
